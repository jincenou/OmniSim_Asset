"""Frozen-anchor actual Adam parameter-update calibration; training views only."""
import argparse,copy,math,os
from quality_common import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--label',choices=['N04','N11'],required=True);args=ap.parse_args()
    start=time.monotonic();spec=plan();device=verify();sys.addaudithook(protect_inputs)
    previous=json.loads((PREVIOUS/'PLAN.json').read_text());item=previous['identities'][args.label];source=Path(item['start_run'])
    assert sha(source/'recovery.pt')==item['files']['recovery.pt']['sha256']
    model,deform,pipe,config,data,state=restore(source)
    lrs=json.loads(Path(item['lr']['path']).read_text());assert sha(item['lr']['path'])==item['lr']['sha256']
    optimizers=[model.optimizer,deform.optimizer]
    for opt in optimizers:
        for g in opt.param_groups:g['lr']=lrs[0][g['name']]
    params=[p for o in optimizers for g in o.param_groups for p in g['params']]
    names=[g['name'] for o in optimizers for g in o.param_groups for p in g['params']]
    mapping={'scaling':'scale','f_dc':'SH','f_rest':'SH','grid':'hash'}
    report_names=[mapping.get(n,n) for n in names]
    train=json.loads(Path(item['data']['training']).read_text());paths={(r['view'],r['frame']):r['path'] for r in train}
    anchors=[a for a in json.loads(Path(spec['calibration']['anchors']).read_text()) if a['label']==args.label]
    observations=[]
    for a in anchors:
        v,f=a['view'],a['frame'];assert v in data['train_views']
        with Image.open(paths[v,f]) as im:rgba=torch.tensor(np.asarray(im).copy(),device='cuda',dtype=torch.float32).permute(2,0,1)/255
        target=rgba[:3]*rgba[3:]+.5*(1-rgba[3:]);partition=torch.tensor(np.load(PREVIOUS/'data'/args.label/'bands'/f'view{v:02d}.npy',mmap_mode='r')[f].copy(),device='cuda')
        observations.append((a,camera(data,v,f,target),rgba[3],partition))
    optconf=SimpleNamespace(**config['optimization']);renders=backwards=simulations=0
    def gradients(obs,arm,coefficient):
        nonlocal renders,backwards
        a,cam,target,partition=obs;set_rng(state['rng'])
        warp=deform.step(model.get_xyz.detach(),cam.fid.unsqueeze(0).expand(len(model.get_xyz),-1))
        rgb=render(cam,model,pipe,torch.full((3,),.5,device='cuda'),warp['d_xyz'],warp['d_rotation'],warp['d_scaling'])['render']
        alpha=render(cam,model,pipe,torch.zeros(3,device='cuda'),warp['d_xyz'],warp['d_rotation'],warp['d_scaling'],override_color=torch.ones_like(model.get_xyz))['render'][0]
        extra,valid=extra_alpha(alpha,target,partition,arm);assert valid
        reg=warp['reg_spatial'].mean()*optconf.lambda_spatial_tv+warp['reg_temporal'].mean()*optconf.lambda_temporal_tv
        loss=.8*(rgb-cam.original_image).abs().mean()+.2*(1-ssim(rgb,cam.original_image))+.1*old_alpha(alpha,target)+reg+coefficient*extra
        grads=torch.autograd.grad(loss,params,allow_unused=True);renders+=2;backwards+=1
        assert all(bool(torch.isfinite(g).all()) for g in grads if g is not None)
        return grads
    def actual_adam_delta(grads):
        nonlocal simulations
        out=[];offset=0
        for optimizer in optimizers:
            groups=[];pairs=[]
            for group in optimizer.param_groups:
                clones=[]
                for p in group['params']:
                    q=torch.nn.Parameter(p.detach().clone());g=grads[offset];offset+=1
                    q.grad=None if g is None else g.detach().clone();clones.append(q);pairs.append((p,q))
                groups.append(dict(group,params=clones))
            simulated=torch.optim.Adam(groups,**optimizer.defaults)
            for p,q in pairs:
                simulated.state[q]={k:v.clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in optimizer.state[p].items()}
            simulated.step()
            out.extend((q.detach()-p.detach()).cpu() for p,q in pairs)
            del simulated,groups,pairs
        simulations+=1
        return out
    references=[]
    for obs in observations:
        da=actual_adam_delta(gradients(obs,'A',0.));du=actual_adam_delta(gradients(obs,'U',.1))
        references.append((da,[u-a for u,a in zip(du,da)]))
    def compare(db,du):
        groups={}
        for name,b,u in zip(report_names,db,du):
            entry=groups.setdefault(name,[0.,0.,0.,0]);entry[0]+=float(b.double().square().sum());entry[1]+=float(u.double().square().sum());entry[2]+=float((b.double()*u.double()).sum());entry[3]+=u.numel()
        groups['geometry_combined_xyz_scale_rotation']=[sum(groups[n][j] for n in ['xyz','scale','rotation']) for j in range(4)]
        result={}
        for name,(bb,uu,bu,count) in groups.items():
            valid=math.sqrt(uu/count)>1e-12 and all(math.isfinite(v) for v in [bb,uu,bu])
            result[name]=dict(dB_l2=math.sqrt(bb),dU_l2=math.sqrt(uu),dU_rms=math.sqrt(uu/count),ratio=math.sqrt(bb/uu) if valid else None,cosine=bu/math.sqrt(bb*uu) if valid and bb>0 else None,parameters=count,valid=valid)
        return result
    destination=NEW/'research/update_calibration'/f'{args.label}.json';assert not destination.exists();evaluations=[]
    def evaluate(coefficient):
        rows=[]
        for obs,(da,du) in zip(observations,references):
            step=actual_adam_delta(gradients(obs,'B',coefficient));db=[b-a for b,a in zip(step,da)]
            rows.append(dict(anchor=obs[0]['id'],groups=compare(db,du)))
        medians={};valid=True
        for group in spec['calibration']['primary_groups']:
            ratios=[r['groups'][group]['ratio'] for r in rows if r['groups'][group]['valid']]
            medians[group]=float(np.median(ratios)) if len(ratios)>=4 else None
            valid=valid and len(ratios)>=4
        objective=float(np.mean([math.log(max(v,1e-300))**2 for v in medians.values()])) if valid else None
        row=dict(coefficient=coefficient,anchors=rows,medians=medians,objective=objective,matched=valid and all(.8<=v<=1.25 for v in medians.values()))
        evaluations.append(row);save(destination.with_name(args.label+'_progress.json'),dict(state='calibrating',device=device,evaluations=evaluations,seconds=time.monotonic()-start));print(args.label,coefficient,medians,flush=True)
        return objective if objective is not None else float('inf')
    first=evaluate(.1)
    if not evaluations[0]['matched']:
        evaluate(1e-5);left,right=math.log(1e-5),math.log(.1);ratio=(math.sqrt(5)-1)/2
        x1=right-ratio*(right-left);x2=left+ratio*(right-left);f1=evaluate(math.exp(x1));f2=evaluate(math.exp(x2))
        while len(evaluations)<12:
            if f1<f2:
                right,x2,f2=x2,x1,f1;x1=right-ratio*(right-left);f1=evaluate(math.exp(x1))
            else:
                left,x1,f1=x1,x2,f2;x2=left+ratio*(right-left);f2=evaluate(math.exp(x2))
    chosen=min(evaluations,key=lambda e:float('inf') if e['objective'] is None else e['objective'])
    trigger=chosen['matched'] and not evaluations[0]['matched']
    for name in PARAMETERS:equal(getattr(model,name),state['gaussians'][name])
    equal(deform.grid4d.state_dict(),state['hash']);equal(deform.deform.state_dict(),state['deform'])
    # Only LR values intentionally changed in original optimizer; moments remain exact.
    for optimizer,saved in zip(optimizers,[state['gaussian_optimizer'],state['deform_optimizer']]):equal(optimizer.state_dict()['state'],saved['state'])
    save(destination,dict(state='passed',label=args.label,device=device,pid=os.getpid(),source=str(source),source_sha256=sha(source/'recovery.pt'),first_A_lr=lrs[0],simulation='Actual torch.optim.Adam on cloned float32 parameters and exact moments, original defaults/bias correction/eps, no source updates',evaluations=evaluations,selected=chosen,trigger_Bmatch=trigger,reason='original_.1_already_matched' if evaluations[0]['matched'] else 'matched_changed_coefficient' if trigger else 'no_scalar_matched_all_primary_groups',source_parameters_and_moments_unchanged=True,costs=dict(seconds=time.monotonic()-start,render_calls=renders,backward_calls=backwards,simulated_optimizer_updates=simulations,source_updates=0),quality_plan_sha256=sha(NEW/'PLAN.json')))
    print(args.label,'calibration complete, trigger=',trigger,flush=True)
if __name__=='__main__':main()
