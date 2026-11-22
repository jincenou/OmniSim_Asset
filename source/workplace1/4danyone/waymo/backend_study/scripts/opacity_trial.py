"""Frozen-geometry, equal-update opacity diagnostic with matched RGB control."""
import argparse,gc,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from common import ROOT,BASE,manifest,camera,device,save,sha,stamp,image_bytes
from evaluate_case import CEM,render,native_ssim,scalar,byte,writer_class,KEYS
DIAG=ROOT/'diagnostics/opacity'

def tensor_hash(values):
    h=hashlib.sha256()
    for name,tensor in values:
        h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def train(label,condition):
    begin=time.monotonic();gpu=device();plan=json.loads((DIAG/'PLAN.json').read_text())
    check=json.loads((ROOT/'research/alpha_gradient_probe.json').read_text())
    assert check['unit_rgb_backward_check']=='passed'
    root=DIAG/'runs'/label/condition;root.mkdir(parents=True,exist_ok=False)
    model=CEM(label,2);g=model.model
    assert sha(model.root/'asset.pt')==plan['sources'][label]
    cfg=dict(model.config);cfg.update(kind='opacity_only_development_diagnostic',diagnostic=plan,condition=condition)
    save(root/'config.json',cfg)
    frozen=[]
    for group in g.optimizer.param_groups:
        for index,p in enumerate(group['params']):
            p.requires_grad_(group['name']=='opacity')
            if group['name']!='opacity':frozen.append((f"{group['name']}/{index}",p))
    assert g._opacity.requires_grad
    frozen.extend([('parent',g._parent),('level',g._level)])
    before=tensor_hash(frozen)
    data,_=manifest(label);rng=np.random.default_rng(plan['seed'])
    trace=np.column_stack([rng.choice(data['train_views'],1000),rng.integers(0,data['frame_count'],1000)])
    np.save(root/'sampling.npy',trace)
    ssim=native_ssim();weight=plan['conditions'][condition]['mask_weight']
    black=torch.zeros(3,device='cuda');white=torch.ones(3,device='cuda')
    unit=torch.ones_like(g._xyz);losses=[]
    for iteration,(view,frame) in enumerate(trace,1):
        cam=camera(data,int(view),int(frame))
        rgba=image_bytes(cam.image_path).to('cuda',dtype=torch.float32)/255.
        target=rgba[:3]*rgba[3:]+1-rgba[3:]
        g.optimizer.zero_grad(set_to_none=True)
        result=render(cam,g,model.pipe,white,near=.01,far=100.)
        alpha=render(cam,g,model.pipe,black,override_color=unit,near=.01,far=100.)['render'][:1]
        if iteration in [1,1000]:assert float((alpha.detach()-result['acc'].detach()).abs().max())<1e-5
        rgb_loss=.8*(result['render']-target).abs().mean()+.2*(1-ssim(result['render'][None],target[None]))
        alpha_loss=(alpha-rgba[3:]).abs().mean()
        loss=rgb_loss+weight*alpha_loss
        loss.backward();assert torch.isfinite(g._opacity.grad).all()
        g.optimizer.step();assert torch.isfinite(g._opacity).all()
        if iteration in [1,100,500,1000]:
            row=dict(iteration=iteration,rgb_loss=float(rgb_loss),alpha_loss=float(alpha_loss),loss=float(loss))
            losses.append(row);save(root/'progress.json',dict(label=label,condition=condition,device=gpu,updated_at=stamp(),**row))
    assert before==tensor_hash(frozen),'Non-opacity parameters changed'
    del result,alpha,loss,rgb_loss,alpha_loss
    torch.save((g.capture(),36000),root/'asset.pt')
    probes={}
    with torch.no_grad():
        for frame in [0,(data['frame_count']-1)//2,data['frame_count']-1]:
            for view in [0,8,16]:
                raw=render(camera(data,view,frame),g,model.pipe,black,near=.01,far=100.)
                probes[f'rgb_{view}_{frame}']=raw['render'].cpu().numpy()
                probes[f'alpha_{view}_{frame}']=raw['acc'].cpu().numpy()
    np.savez_compressed(root/'replay_reference.npz',**probes)
    save(root/'complete.json',dict(label=label,condition=condition,device=gpu,ended_at=stamp(),
        asset_sha256=sha(root/'asset.pt'),actual_extra_updates=1000,frozen_parameters_sha256=before,
        source_asset_sha256=plan['sources'][label],sampling_sha256=sha(root/'sampling.npy'),
        elapsed_seconds=time.monotonic()-begin,losses=losses,production_approved=False))

@torch.inference_mode()
def evaluate(label):
    import cv2,lpips
    gpu=device();final=DIAG/'results'/label
    assert not final.exists()
    out=DIAG/'tmp'/label;out.mkdir(parents=True,exist_ok=False)
    data,_=manifest(label);frames=data['frame_count']
    assert sha(DIAG/'runs'/label/'control/sampling.npy')==sha(DIAG/'runs'/label/'mask/sampling.npy')
    models={'cem35k':CEM(label,2),'control':CEM(label,2,root=DIAG/'runs'/label/'control'),
            'mask':CEM(label,2,root=DIAG/'runs'/label/'mask')}
    for model in models.values():model.validate()
    prior=json.loads((ROOT/'results'/label/'results.json').read_text())
    baseline=json.loads((BASE/'samples'/label/'evaluations/4dgs_grid61/results.json').read_text())
    sample=set(baseline['sampled_frames']);crops={int(k):v for k,v in baseline['crops'].items()}
    ssim=native_ssim();lp=lpips.LPIPS(net='alex').cuda().eval()
    rows={m:[] for m in models};temporal={m:[] for m in models};video_view=0 if label=='N04' else 8
    Writer=writer_class();last={};max_error={k:0. for k in KEYS}
    with Writer(str(out/'comparison.mp4'),25,1408,672) as writer:
        for view in [0,8,16]:
            last={}
            for frame in range(frames):
                p=Path(data['source_data'])/'images'/f'cam{view:02d}'/f'{frame:06d}.png'
                im=np.asarray(Image.open(p),np.float32)/255.;mask=im[...,3]>.5
                ref=im[...,:3]*im[...,3:]+(1-im[...,3:])*.5;preds={}
                for name,model in models.items():
                    rgb,alpha=model.raw(view,frame);pred=np.clip(rgb+(1-alpha[...,None])*.5,0,1);preds[name]=pred
                    if last:
                        residual=(pred-last['predictions'][name])-(ref-last['reference']);union=mask|last['reference_mask']
                        temporal[name].append(float((residual[union]**2).mean()))
                    if frame in sample:
                        row=scalar(pred,ref,alpha,mask);x0,y0,x1,y1=crops[view]
                        a=torch.tensor(pred.transpose(2,0,1).copy(),device='cuda')[None]
                        b=torch.tensor(ref.transpose(2,0,1).copy(),device='cuda')[None]
                        row.update(view=view,frame=frame,ssim=float(ssim(a,b)),
                            crop_lpips=float(lp(a[:,:,y0:y1,x0:x1]*2-1,b[:,:,y0:y1,x0:x1]*2-1)))
                        rows[name].append(row)
                        if name=='cem35k':
                            old=next(r for r in prior['samples']['cem_stage2'] if r['view']==view and r['frame']==frame)
                            for key in KEYS:max_error[key]=max(max_error[key],abs(row[key]-old[key]))
                    assert np.isfinite(pred).all()
                if view==video_view:
                    canvas=Image.new('RGB',(1408,672),(24,24,24))
                    canvas.paste(Image.fromarray(byte(np.concatenate([ref,*preds.values()],axis=1))),(0,32))
                    draw=ImageDraw.Draw(canvas)
                    for col,title in enumerate(['Generated reference','CEM complete35k','+1000 RGB / opacity only','+1000 RGB+mask / opacity only']):draw.text((352*col+6,9),title,fill='white')
                    writer.append_data(np.asarray(canvas))
                    if frame in [0,(frames-1)//2,frames-1]:canvas.save(out/f'frame{frame:03d}.jpg',quality=95)
                last={'reference':ref,'reference_mask':mask,'predictions':preds}
    assert max(max_error.values())<1e-4,max_error
    cap=cv2.VideoCapture(str(out/'comparison.mp4'));count=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape==(672,1408,3);count+=1
    cap.release();assert count==frames
    save(out/'results.json',dict(label=label,state='evaluated_visual_review_pending',updated_at=stamp(),device=gpu,
        mean={m:{k:float(np.mean([r[k] for r in rr])) for k in KEYS} for m,rr in rows.items()},samples=rows,
        temporal_fg_union_residual_mse={m:float(np.mean(v)) for m,v in temporal.items()},
        original_metric_max_error=max_error,replay={m:v.replay for m,v in models.items()},
        video=dict(view=video_view,frames=count,sha256=sha(out/'comparison.mp4')),
        protocol=prior['protocol'],scope='1000 extra opacity-only updates; control differs only in mask weight. Development diagnosis, no geometry update, not full CEM retraining.',production_approved=False))
    final.parent.mkdir(parents=True,exist_ok=True)
    out.rename(final)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',choices=['N04','N08'],required=True)
    p.add_argument('--condition',choices=['control','mask','evaluate'],required=True);args=p.parse_args()
    if args.condition=='evaluate':evaluate(args.label)
    else:train(args.label,args.condition)
