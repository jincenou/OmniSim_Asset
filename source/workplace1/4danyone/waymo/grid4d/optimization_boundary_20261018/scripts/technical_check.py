"""Exact recovery, explicit region math and native boundary contribution checks."""
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import torch
from PIL import Image
from boundary_common import *
from boundary_loss import DISK


def weights_check():
    soft=np.zeros((17,19),np.float64);soft[:11,:12]=.85;soft[10,:12]=.6;soft[11,:12]=.2
    result=regions(soft);mask=soft>=.5;pad=np.pad(mask,3);erode=np.ones_like(mask);dilate=np.zeros_like(mask)
    for y,x in np.argwhere(DISK):
        erode &= pad[y:y+17,x:x+19];dilate |= pad[y:y+17,x:x+19]
    assert np.array_equal(result==1,erode) and np.array_equal(result==2,mask&~erode) and np.array_equal(result==3,dilate&~mask)
    target=torch.tensor(soft);part=torch.tensor(result);pred=(target*.6+.17).requires_grad_();rows=[]
    for arm in ['U','B']:
        loss,valid=extra_alpha(pred,target,part,arm);assert valid
        positive=target>=.5 if arm=='U' else part==2;negative=target<.5 if arm=='U' else part==3
        weight=.5*positive/positive.sum()+.5*negative/negative.sum();assert abs(float(weight.sum())-1)<1e-7
        explicit=(weight*(pred-target).abs()).sum();assert abs(float(loss-explicit))<1e-8
        actual=torch.autograd.grad(loss,pred,retain_graph=True)[0];wanted=weight*torch.sign(pred-target)
        error=float((actual-wanted).abs().max());assert error<1e-8
        direction=torch.linspace(-.2,.2,pred.numel()).reshape(pred.shape);eps=1e-5
        plus=extra_alpha(pred+eps*direction,target,part,arm)[0];minus=extra_alpha(pred-eps*direction,target,part,arm)[0]
        analytic=float((actual*direction).sum());numeric=float((plus-minus)/(2*eps));assert abs(analytic-numeric)<1e-8
        rows.append(dict(arm=arm,weight_mass=float(weight.sum()),positive_pixels=int(positive.sum()),negative_pixels=int(negative.sum()),gradient_max_error=error,finite_difference_error=abs(analytic-numeric)))
    empty=[]
    for value in [0.,1.]:
        ref=torch.full((17,19),value);partition=torch.tensor(regions(ref.numpy()));prediction=torch.full_like(ref,.3,requires_grad=True)
        for arm in ['U','B']:
            loss,valid=extra_alpha(prediction,ref,partition,arm);assert not valid and float(loss)==0
            assert float(torch.autograd.grad(loss,prediction)[0].abs().sum())==0;empty.append(dict(value=value,arm=arm,skipped=True))
    return dict(state='passed',disk=DISK.tolist(),explicit_zero_padding_equal=True,checks=rows,empty_policy=empty)


def captured_output(call):
    import diff_gaussian_rasterization as native
    original=native._C.rasterize_gaussians;record={}
    def observe(*args):
        out=original(*args)
        record.update(count=int(out[0]),radii=out[3].detach().cpu().numpy(),geometry=out[4].detach().cpu().numpy(),binning=out[5].detach().cpu().numpy(),image=out[6].detach().cpu().numpy())
        assert all(v.data_ptr()%128==0 for v in out[4:7])
        return out
    native._C.rasterize_gaussians=observe
    try:result=call()
    finally:native._C.rasterize_gaussians=original
    return result,record


def contributors(buffers,n,y,x):
    # Read the pinned implementation's aligned forward buffers, without changing CUDA code.
    offset=0
    def get(dtype,count):
        nonlocal offset
        offset=(offset+127)//128*128;out=np.frombuffer(buffers['geometry'],dtype=dtype,count=count,offset=offset);offset+=np.dtype(dtype).itemsize*count;return out
    get(np.float32,n);get(np.bool_,n*3);get(np.int32,n);xy=get(np.float32,n*2).reshape(n,2);get(np.float32,n*6);conic=get(np.float32,n*4).reshape(n,4)
    pixels=352*640;one=(pixels*4+127)//128*128;two=(one+pixels*4+127)//128*128
    ranges=np.frombuffer(buffers['image'],np.uint32,count=pixels*2,offset=two).reshape(pixels,2)
    last=np.frombuffer(buffers['image'],np.uint32,count=pixels,offset=one)[y*352+x]
    expected_T=np.frombuffer(buffers['image'],np.float32,count=pixels)[y*352+x]
    tile=(y//16)*((352+15)//16)+x//16;begin,end=ranges[tile];ids=np.frombuffer(buffers['binning'],np.uint32,count=buffers['count'])[begin:end]
    d=xy[ids]-np.asarray([x,y],np.float32);c=conic[ids];power=-.5*(c[:,0]*d[:,0]**2+c[:,2]*d[:,1]**2)-c[:,1]*d[:,0]*d[:,1]
    a=np.minimum(.99,c[:,3]*np.exp(np.minimum(power,0)));active=(power<=0)&(a>=1/255)
    # Last contributor is the forward traversal index, not a count of accepted points.
    active[np.arange(len(ids))>=last]=False
    selected=ids[active];T=float(np.prod(1-a[active],dtype=np.float64));assert abs(T-float(expected_T))<1e-5,(T,expected_T)
    return dict(ids=selected.tolist(),sha256=hashlib.sha256(selected.tobytes()).hexdigest(),last=int(last),T=T,buffer_T=float(expected_T),minimum_threshold_distance=float(np.min(np.abs(a-1/255))) if len(a) else None)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);args=parser.parse_args();begin=time.monotonic();device=verify();sys.addaudithook(protect_inputs)
    report=dict(label=args.label,device=device,state='running',weights=weights_check(),plan_sha256=sha(NEW/'PLAN.json'))
    model,deform,pipe,config,data,state=restore(args.label,regularize=False)
    report['restore']=dict(all_gaussian_hash_deform_adam_rng_equal=True,groups=plan()['identities'][args.label]['optimizer_groups'],replay=check_start_replay(args.label,model,deform,pipe,data))
    anchors=json.loads((NEW/'data/anchors.json').read_text());anchor=next(a for a in anchors if a['label']==args.label)
    view,frame=anchor['view'],anchor['frame'];paths=json.loads(Path(plan()['identities'][args.label]['data']['training']).read_text());row=next(r for r in paths if r['view']==view and r['frame']==frame)
    with Image.open(row['path']) as image:target=torch.tensor(np.asarray(image)[...,3].copy(),device='cuda',dtype=torch.float32)/255
    partition=torch.tensor(regions(target.cpu().numpy()),device='cuda');cam=camera(data,view,frame)
    with torch.no_grad():warp=deform.step(model.get_xyz.detach(),cam.fid.unsqueeze(0).expand(len(model.get_xyz),-1))
    def output():return render(cam,model,pipe,torch.zeros(3,device='cuda'),warp['d_xyz'],warp['d_rotation'],warp['d_scaling'],override_color=torch.ones_like(model.get_xyz))['render'][0]
    alpha,base=captured_output(output)
    band=((partition==2)|(partition==3))&(alpha>.05)&(alpha<.9)&((alpha-target).abs()>.02)
    locations=torch.nonzero(band).cpu().numpy();assert len(locations)>=4
    chosen=locations[np.rint(np.linspace(0,len(locations)-1,4)).astype(int)];rows=[];initial=model._opacity.detach().clone();epsilon=.001
    for y,x in chosen:
        value=(alpha[y,x]-target[y,x]).abs();grad=torch.autograd.grad(value,model._opacity,retain_graph=True)[0].flatten();point=int(grad.abs().argmax());analytic=float(grad[point]);baseline=contributors(base,len(model.get_xyz),int(y),int(x));assert point in baseline['ids']
        with torch.no_grad():
            model._opacity[point]=initial[point]+epsilon;plus,pbuf=captured_output(output)
            model._opacity[point]=initial[point]-epsilon;minus,mbuf=captured_output(output)
            model._opacity.copy_(initial)
        pset=contributors(pbuf,len(model.get_xyz),int(y),int(x));mset=contributors(mbuf,len(model.get_xyz),int(y),int(x))
        stable=baseline['ids']==pset['ids']==mset['ids'];numeric=float(((plus[y,x]-target[y,x]).abs()-(minus[y,x]-target[y,x]).abs())/(2*epsilon))
        zero=abs(analytic)<=1e-7 and abs(numeric)<=1e-7;relative=abs(analytic-numeric)/max(abs(numeric),1e-12)
        rows.append(dict(view=view,frame=frame,y=int(y),x=int(x),region=int(partition[y,x]),point=point,prediction=float(alpha[y,x]),reference=float(target[y,x]),analytic=analytic,numeric=numeric,relative_error=relative,epsilon=epsilon,stable_contributor_set=stable,zero_gradient=zero,contribution_count=len(baseline['ids']),contributor_sha256=baseline['sha256'],minimum_cutoff_distance=baseline['minimum_threshold_distance'],pass_stable_nonzero=stable and not zero and relative<.01))
    report['boundary_piecewise']=rows
    # Retain the full-band directional check even when cutoff switches make it fail.
    global_band=extra_alpha(alpha.double(),target.double(),partition,'B')[0];g=torch.autograd.grad(global_band,model._opacity)[0];analytic=float(g.sum())
    with torch.no_grad():
        model._opacity.copy_(initial+epsilon);plus=extra_alpha(output().double(),target.double(),partition,'B')[0]
        model._opacity.copy_(initial-epsilon);minus=extra_alpha(output().double(),target.double(),partition,'B')[0]
        model._opacity.copy_(initial)
    numeric=float((plus-minus)/(2*epsilon));relative=abs(analytic-numeric)/max(abs(numeric),1e-12)
    report['full_band_directional']=dict(analytic=analytic,numeric=numeric,relative_error=relative,within_original_threshold=relative<.01,contribution_stability='not assumed; all opacity logits perturbed; does not override stable-piece checks')
    report['old_failure_preserved']=str(ROOT/'optimization/alpha_gradient_diagnosis.json')
    report['scope']='Exact scalar region loss plus four actual reference-boundary pixels, original native renderer and unchanged threshold; does not prove global differentiability'
    report['seconds']=time.monotonic()-begin;report['state']='passed' if all(r['pass_stable_nonzero'] for r in rows) else 'failed_boundary_probe'
    save(NEW/'research'/f'{args.label}_technical.json',report);assert report['state']=='passed',rows
    print(args.label,'exact recovery and boundary checks passed',flush=True)


if __name__=='__main__':main()
