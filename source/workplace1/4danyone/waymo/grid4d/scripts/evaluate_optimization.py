"""Independent fixed-final evaluation of the frozen 2x4 continuation experiment."""
import argparse
import json
import numpy as np
from PIL import Image
import torch
import lpips
from grid_common import ROOT, BASE, load_asset, raw_render, camera, manifest, sha, save, verify, stamp
from evaluate_grid import frozen_functions


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);parser.add_argument('--arm',required=True)
    args=parser.parse_args();device=verify();key=f'{args.label}_{args.arm}'
    plan=json.loads((ROOT/'optimization/PLAN.json').read_text())
    assert args.label in plan['labels'] and args.arm in plan['arms']
    run=ROOT/'optimization/runs'/key;iteration=48000
    complete=json.loads((run/'complete.json').read_text());assert complete['extra_updates']==15000
    validation=json.loads((run/'validation/48000.json').read_text());assert validation['state']=='passed'
    snapshot=json.loads((run/'snapshots/48000.json').read_text())
    assert snapshot['actual_optimizer_updates']==47999
    for file,digest in snapshot['files'].items():assert sha(run/file)==digest
    gs,deform,pipe,config=load_asset(run,iteration);data,_=manifest(args.label)
    camera_data=json.loads((run/'asset_camera_data.json').read_text())
    scalar,crop,(ssim,ssim_sha)=frozen_functions();perceptual=lpips.LPIPS(net='alex').cuda().eval()
    sampled=np.rint(np.linspace(0,data['frame_count']-1,13)).astype(int).tolist()
    out=ROOT/'optimization/results'/key;out.mkdir(parents=True,exist_ok=False)
    cache=ROOT/'optimization/raw'/key;cache.mkdir(parents=True,exist_ok=False)
    rows=[];dense=[];crops={};files={}
    for view in [0,8,16]:
        def target_path(frame):return BASE/'samples'/args.label/'data/images'/f'cam{view:02d}'/f'{frame:06d}.png'
        masks=[np.asarray(Image.open(target_path(frame)))[...,3]>127 for frame in sampled]
        box=crop(masks);crops[view]=box;x0,y0,x1,y1=box;rgbs=[];alphas=[]
        for frame in range(data['frame_count']):
            rgba=np.asarray(Image.open(target_path(frame)),np.float32)/255
            target=rgba[...,:3]*rgba[...,3:]+.5*(1-rgba[...,3:]);mask=rgba[...,3]>.5
            rgb,alpha=raw_render(gs,deform,pipe,camera(camera_data,view,frame))
            rgb=rgb.permute(1,2,0).cpu().numpy();alpha=alpha[0].cpu().numpy()
            assert np.isfinite(rgb).all() and np.isfinite(alpha).all()
            assert alpha.min()>=-1e-6 and alpha.max()<=1.000001
            pred=np.clip(rgb+.5*(1-alpha[...,None]),0,1)
            values=dict(view=view,frame=frame,seconds=frame/25,**scalar(pred,target,alpha,mask))
            values['foreground_soft_coverage']=float(alpha[mask].mean());dense.append(values.copy())
            if frame in sampled:
                a=torch.as_tensor(pred.transpose(2,0,1).copy(),device='cuda')[None]
                b=torch.as_tensor(target.transpose(2,0,1).copy(),device='cuda')[None]
                values.update(ssim=float(ssim(a,b)),crop_lpips=float(perceptual(a[:,:,y0:y1,x0:x1]*2-1,b[:,:,y0:y1,x0:x1]*2-1)),crop_xyxy=box,input_sha256=sha(target_path(frame)))
                rows.append(values)
            rgbs.append(rgb);alphas.append(alpha)
        file=cache/f'view{view:02d}.npz';np.savez_compressed(file,rgb=np.stack(rgbs),alpha=np.stack(alphas));files[file.name]=sha(file)
    assert len(rows)==39
    baseline=json.loads((ROOT/'results'/args.label/'33000/metrics.json').read_text())
    assert [(r['view'],r['frame'],r['crop_xyxy'],r['input_sha256']) for r in rows]==[(r['view'],r['frame'],r['crop_xyxy'],r['input_sha256']) for r in baseline['samples']]
    fields=['full_psnr','foreground_psnr','ssim','crop_lpips','silhouette_iou','outside_alpha_mean','foreground_soft_coverage']
    report=dict(state='complete_fixed_final_metrics_visual_review_pending',label=args.label,arm=args.arm,iteration=iteration,actual_optimizer_updates=47999,extra_updates=15000,device=device,at=stamp(),
        mean={k:float(np.mean([r[k] for r in rows])) for k in fields},samples=rows,dense_samples=dense,
        protocol=dict(name='random8-v1.1-generated-heldout',heldout_views=[0,8,16],sampled_frames=sampled,crops=crops,background=.5,resolution=[352,640],ssim_source_sha256=ssim_sha,lpips='lpips0.1.4 AlexNet v0.1; [-1,1]',reference='Frozen generated development reference; not real Waymo novel-view truth',fixed_final_no_best_checkpoint_selection=True,exact_reference_denominators_and_crops_equal_baseline=True,same_gpu_compute_budget=False),
        asset=dict(run=str(run),snapshot=str(run/'snapshots/48000.json'),validation=str(run/'validation/48000.json'),config=str(run/'config.json')),training=complete)
    # Exposure ledger is retained in the run; avoid duplicating it in every result.
    report['training']={k:v for k,v in complete.items() if k!='continuation_exposure'}
    save(out/'metrics.json',report);save(cache/'complete.json',dict(state='complete',views=[0,8,16],frames=data['frame_count'],files=files,source_results=str(out/'metrics.json')))
    print(json.dumps(dict(key=key,mean=report['mean'])),flush=True)


if __name__=='__main__':main()
