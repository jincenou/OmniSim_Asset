"""Frozen H global metrics; terminal complete-time render cache for all local tests."""
from export import *
from evaluate_grid import frozen_functions
import lpips,csv,gc

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--key',required=True);ap.add_argument('--step',required=True,type=int);args=ap.parse_args();begin=time.monotonic();device=verify();label=args.key[:3];key=f'{args.key}_{args.step}';run=NEW/'runs'/args.key
    out=NEW/'results'/key;out.mkdir(parents=True,exist_ok=True);assert not (out/'metrics.json').exists();dense=args.step==32999
    if dense:
        asset=NEW/'assets'/args.key;assert json.loads((asset/'replay_validation.json').read_text())['passed'];g,d,q,z,cfg,data,meta=load_portable(asset)
    else:
        g,d,q,cfg,data,_=load_recovery(run/f'recovery_{args.step}.pt');inp=Inputs(label,data,False);z=bake(q,d,g,data,inp,args.step);del inp;torch.cuda.empty_cache()
    pipe=SimpleNamespace(**cfg['pipeline']);scalar,crop,(ssim,ssim_sha)=frozen_functions();perceptual=lpips.LPIPS(net='alex').cuda().eval();sampled=np.rint(np.linspace(0,data['frame_count']-1,13)).astype(int).tolist();rows=[];dense_rows=[];crops={};files={}
    cache=NEW/'cache/raw'/key
    if dense:cache.mkdir(parents=True,exist_ok=False)
    for v in [0,8,16]:
        def refpath(f):return BASE/'samples'/label/'data/images'/f'cam{v:02d}'/f'{f:06d}.png'
        box=crop([np.asarray(Image.open(refpath(f)))[...,3]>127 for f in sampled]);crops[v]=box;x0,y0,x1,y1=box;rgbs=[];alphas=[]
        for f in range(data['frame_count']) if dense else sampled:
            rgba=np.asarray(Image.open(refpath(f)),np.float32)/255;target=rgba[...,:3]*rgba[...,3:]+.5*(1-rgba[...,3:]);mask=rgba[...,3]>.5
            state=geometry(g,d,q,data,f,args.step,baked=z);cam=camera(data,v,f);raw=raster(g,pipe,cam,state,0.)['render'].permute(1,2,0).cpu().numpy();alpha=raster(g,pipe,cam,state,0.,True)['render'][0].cpu().numpy();assert np.isfinite(raw).all() and np.isfinite(alpha).all()
            pred=np.clip(raw+.5*(1-alpha[...,None]),0,1);values=dict(view=v,frame=f,seconds=f/25,**scalar(pred,target,alpha,mask),foreground_soft_coverage=float(alpha[mask].mean()))
            if dense:rgbs.append(raw);alphas.append(alpha);dense_rows.append(values.copy())
            if f in sampled:
                a=torch.as_tensor(pred.transpose(2,0,1).copy(),device='cuda')[None];b=torch.as_tensor(target.transpose(2,0,1).copy(),device='cuda')[None]
                values.update(ssim=float(ssim(a,b)),crop_lpips=float(perceptual(a[:,:,y0:y1,x0:x1]*2-1,b[:,:,y0:y1,x0:x1]*2-1)),crop_xyxy=box,input_sha256=sha(refpath(f)));rows.append(values)
        if dense:
            p=cache/f'view{v:02d}.npz';np.savez_compressed(p,rgb=np.stack(rgbs),alpha=np.stack(alphas));files[p.name]=sha(p)
    baseline=json.loads((BASE/'samples'/label/'evaluations/4dgs_grid61/results.json').read_text());assert len(rows)==39
    assert sorted((r['view'],r['frame'],r['crop_xyxy'],r['input_sha256']) for r in rows)==sorted((r['view'],r['frame'],r['crop_xyxy'],r['input_sha256']) for r in baseline['samples'])
    fields=['full_psnr','foreground_psnr','ssim','crop_lpips','silhouette_iou','outside_alpha_mean','foreground_soft_coverage']
    save(out/'metrics.json',dict(state='metrics_complete_visual_review_pending',label=label,method=key,actual_updates=args.step,device=device,mean={k:float(np.mean([r[k] for r in rows])) for k in fields},samples=rows,dense_samples=dense_rows,protocol=dict(reference='Frozen generated H development reference;not real Waymo hidden-view truth',heldout_views=[0,8,16],resolution=[352,640],background=.5,exact_reference_denominators_and_crops_equal_HUST=True,ssim_source_sha256=ssim_sha,lpips='AlexNet v0.1',paired_new_protocol='same-time2views;not old single-view same compute'),seconds=time.monotonic()-begin))
    if dense:save(cache/'complete.json',dict(state='complete',files=files,frames=data['frame_count'],views=[0,8,16]))
    print(key,'metrics complete',flush=True)
if __name__=='__main__':main()
