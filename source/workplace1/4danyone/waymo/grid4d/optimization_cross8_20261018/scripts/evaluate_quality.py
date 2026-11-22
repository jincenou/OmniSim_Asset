"""Frozen global and optional dense reference metrics for registered snapshots."""
import argparse,csv
import lpips
from quality_common import *
from evaluate_grid import frozen_functions
from region_metrics import rows_for_frame

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--iteration',type=int,required=True);ap.add_argument('--dense',action='store_true');args=ap.parse_args();begin=time.monotonic();device=verify()
    run=Path(args.run);iteration=args.iteration;label=run.name[:3];key=run.name+'_'+str(iteration)
    assert json.loads((run/'validation'/f'{iteration}.json').read_text())['state']=='passed'
    snapshot=json.loads((run/'snapshots'/f'{iteration}.json').read_text());model,deform,pipe,config=load_asset(run,iteration);data=json.loads((run/'asset_camera_data.json').read_text());full,_=manifest(label);source=Path(full['source_data'])
    scalar,crop,(ssim,ssim_sha)=frozen_functions();perceptual=lpips.LPIPS(net='alex').cuda().eval();sampled=np.rint(np.linspace(0,data['frame_count']-1,13)).astype(int).tolist()
    out=NEW/'results'/key;assert not (out/'metrics.json').exists();out.mkdir(parents=True,exist_ok=True);cache=NEW/'cache/raw'/key
    if args.dense:
        assert not (cache/'complete.json').exists();cache.mkdir(parents=True,exist_ok=True)
    rows=[];dense=[];region_rows=[];files={};crops={}
    for view in [0,8,16]:
        def target_path(frame):return source/'images'/f'cam{view:02d}'/f'{frame:06d}.png'
        masks=[np.asarray(Image.open(target_path(frame)))[...,3]>127 for frame in sampled];box=crop(masks);crops[view]=box;x0,y0,x1,y1=box
        cores=np.load(ROOT/'audit/roi'/label/f'view{view:02d}.npz')['labels'] if args.dense else None;rgbs=[];alphas=[]
        for frame in range(data['frame_count']) if args.dense else sampled:
            rgba=np.asarray(Image.open(target_path(frame)),np.float32)/255;target=rgba[...,:3]*rgba[...,3:]+.5*(1-rgba[...,3:]);mask=rgba[...,3]>.5
            rgb,alpha=raw_render(model,deform,pipe,camera(data,view,frame));rgb=rgb.permute(1,2,0).cpu().numpy();alpha=alpha[0].cpu().numpy();assert np.isfinite(rgb).all() and np.isfinite(alpha).all()
            pred=np.clip(rgb+.5*(1-alpha[...,None]),0,1);values=dict(view=view,frame=frame,seconds=frame/25,**scalar(pred,target,alpha,mask),foreground_soft_coverage=float(alpha[mask].mean()))
            if args.dense:
                regions,extra=rows_for_frame(rgb,alpha,rgba,cores[frame],label,key,view,frame,'heldout_development');region_rows.extend(regions);dense.append(dict(**values,**extra));rgbs.append(rgb);alphas.append(alpha)
            if frame in sampled:
                a=torch.as_tensor(pred.transpose(2,0,1).copy(),device='cuda')[None];b=torch.as_tensor(target.transpose(2,0,1).copy(),device='cuda')[None];values.update(ssim=float(ssim(a,b)),crop_lpips=float(perceptual(a[:,:,y0:y1,x0:x1]*2-1,b[:,:,y0:y1,x0:x1]*2-1)),crop_xyxy=box,input_sha256=sha(target_path(frame)));rows.append(values)
        if args.dense:
            path=cache/f'view{view:02d}.npz';np.savez_compressed(path,rgb=np.stack(rgbs),alpha=np.stack(alphas));files[path.name]=sha(path)
    baseline=json.loads((BASE/'samples'/label/'evaluations/4dgs_grid61/results.json').read_text());assert len(rows)==39
    assert sorted((r['view'],r['frame'],r['crop_xyxy'],r['input_sha256']) for r in rows)==sorted((r['view'],r['frame'],r['crop_xyxy'],r['input_sha256']) for r in baseline['samples'])
    fields=['full_psnr','foreground_psnr','ssim','crop_lpips','silhouette_iou','outside_alpha_mean','foreground_soft_coverage']
    save(out/'metrics.json',dict(state='fixed_snapshot_metrics_visual_review_pending',label=label,method=key,run=str(run),iteration=iteration,actual_optimizer_updates=snapshot['actual_optimizer_updates'],color_pass=snapshot.get('color_pass'),color_optimizer_updates=snapshot.get('color_optimizer_updates'),device=device,mean={k:float(np.mean([r[k] for r in rows])) for k in fields},samples=rows,dense_samples=dense,protocol=dict(name='random8-v1.1-generated-heldout',reference='Frozen generated H development references, not real Waymo novel-view truth',heldout_views=[0,8,16],resolution=[352,640],background=.5,crops=crops,sampled_frames=sampled,ssim_source_sha256=ssim_sha,lpips='AlexNet v0.1',fixed_snapshot_no_best_selection=True,exact_reference_denominators_and_crops_equal_HUST=True),seconds=time.monotonic()-begin))
    if args.dense:
        with (out/'regions.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(region_rows[0]));w.writeheader();w.writerows(region_rows)
        save(cache/'complete.json',dict(state='complete',files=files,frames=data['frame_count'],views=[0,8,16],source_results=str(out/'metrics.json')))
    print(key,'evaluation complete',flush=True)
if __name__=='__main__':main()
