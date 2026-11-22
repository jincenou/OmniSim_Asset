"""Fixed final H evaluation, identical primary protocol and independent replay dependency."""
import argparse
import csv
import json
import time
import numpy as np
from PIL import Image
import torch
import lpips
from boundary_common import *
from evaluate_grid import frozen_functions
from region_metrics import rows_for_frame


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);parser.add_argument('--arm',required=True);parser.add_argument('--trial',default='primary');args=parser.parse_args();begin=time.monotonic();device=verify()
    run=NEW/'runs'/args.trial/f'{args.label}_{args.arm}';complete=json.loads((run/'complete.json').read_text());assert complete['total_actual_updates']==52999
    validation=json.loads((run/'validation/53000.json').read_text());assert validation['state']=='passed'
    snapshot=json.loads((run/'snapshots/53000.json').read_text());assert snapshot['actual_optimizer_updates']==52999
    for path,digest in snapshot['files'].items():assert sha(run/path)==digest
    model,deform,pipe,config=load_asset(run,53000);data=json.loads((run/'asset_camera_data.json').read_text())
    spec=plan();train_manifest=json.loads(Path(config['provenance']['manifest']).read_text());source=Path(train_manifest['source_data'])
    scalar,crop,(ssim,ssim_sha)=frozen_functions();perceptual=lpips.LPIPS(net='alex').cuda().eval()
    sampled=np.rint(np.linspace(0,data['frame_count']-1,13)).astype(int).tolist()
    out=NEW/'results'/args.trial/f'{args.label}_{args.arm}';out.mkdir(parents=True,exist_ok=False);cache=NEW/'cache/raw'/args.trial/f'{args.label}_{args.arm}';cache.mkdir(parents=True,exist_ok=False)
    rows=[];dense=[];region_rows=[];crops={};files={}
    for view in [0,8,16]:
        def target_path(frame):return source/'images'/f'cam{view:02d}'/f'{frame:06d}.png'
        masks=[np.asarray(Image.open(target_path(frame)))[...,3]>127 for frame in sampled];box=crop(masks);crops[view]=box;x0,y0,x1,y1=box
        core=np.load(ROOT/'audit/roi'/args.label/f'view{view:02d}.npz')['labels'];rgbs=[];alphas=[]
        for frame in range(data['frame_count']):
            rgba=np.asarray(Image.open(target_path(frame)),np.float32)/255;target=rgba[...,:3]*rgba[...,3:]+.5*(1-rgba[...,3:]);mask=rgba[...,3]>.5
            rgb,alpha=raw_render(model,deform,pipe,camera(data,view,frame));rgb=rgb.permute(1,2,0).cpu().numpy();alpha=alpha[0].cpu().numpy()
            assert np.isfinite(rgb).all() and np.isfinite(alpha).all() and alpha.min()>=-1e-6 and alpha.max()<=1.000001
            pred=np.clip(rgb+.5*(1-alpha[...,None]),0,1);values=dict(view=view,frame=frame,seconds=frame/25,**scalar(pred,target,alpha,mask),foreground_soft_coverage=float(alpha[mask].mean()))
            region,external=rows_for_frame(rgb,alpha,rgba,core[frame],args.label,args.arm,view,frame,'heldout_development');region_rows.extend(region);dense.append(dict(**values,**external))
            if frame in sampled:
                a=torch.as_tensor(pred.transpose(2,0,1).copy(),device='cuda')[None];b=torch.as_tensor(target.transpose(2,0,1).copy(),device='cuda')[None]
                values.update(ssim=float(ssim(a,b)),crop_lpips=float(perceptual(a[:,:,y0:y1,x0:x1]*2-1,b[:,:,y0:y1,x0:x1]*2-1)),crop_xyxy=box,input_sha256=sha(target_path(frame)));rows.append(values)
            rgbs.append(rgb);alphas.append(alpha)
        p=cache/f'view{view:02d}.npz';np.savez_compressed(p,rgb=np.stack(rgbs),alpha=np.stack(alphas));files[p.name]=sha(p)
    baseline=json.loads((ROOT/'results'/args.label/'33000/metrics.json').read_text());assert len(rows)==39
    assert [(r['view'],r['frame'],r['crop_xyxy'],r['input_sha256']) for r in rows]==[(r['view'],r['frame'],r['crop_xyxy'],r['input_sha256']) for r in baseline['samples']]
    fields=['full_psnr','foreground_psnr','ssim','crop_lpips','silhouette_iou','outside_alpha_mean','foreground_soft_coverage']
    save(out/'metrics.json',dict(state='complete_fixed_final_metrics_visual_review_pending',label=args.label,arm=args.arm,trial=args.trial,iteration=53000,actual_optimizer_updates=52999,stage_updates=5000,device=device,mean={k:float(np.mean([r[k] for r in rows])) for k in fields},samples=rows,dense_samples=dense,protocol=dict(name='random8-v1.1-generated-heldout',heldout_views=[0,8,16],sampled_frames=sampled,crops=crops,background=.5,resolution=[352,640],ssim_source_sha256=ssim_sha,lpips='lpips0.1.4 AlexNet v0.1; [-1,1]',reference='Frozen generated development references; not real Waymo novel-view truth',fixed_final_no_best_selection=True,exact_reference_denominators_and_crops_equal_baseline=True),seconds=time.monotonic()-begin,training=complete))
    with (out/'regions.csv').open('x',newline='') as f:writer=csv.DictWriter(f,fieldnames=list(region_rows[0]));writer.writeheader();writer.writerows(region_rows)
    save(cache/'complete.json',dict(state='complete',files=files,frames=data['frame_count'],views=[0,8,16],source_results=str(out/'metrics.json')))
    print(args.label,args.arm,'fixed terminal evaluation complete',flush=True)


if __name__=='__main__':main()
