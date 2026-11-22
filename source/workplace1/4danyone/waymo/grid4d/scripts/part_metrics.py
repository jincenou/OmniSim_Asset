"""Identical reference-core denominators across backends, including complete prediction absence."""
import argparse
import csv
import json
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from prepare_diagnostics import ROOT,BASE,PARTS,remap

BRANCHES=['ftgs_base5k','ftgs_opt15k','4dgs_grid25','4dgs_grid61']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);parser.add_argument('--method',required=True,choices=BRANCHES+['grid4d_33k']+['grid4d_'+x for x in ['continue','parts','alpha','combined']])
    args=parser.parse_args();cv2.setNumThreads(2)
    roi_root=ROOT/'audit/roi'/args.label
    roi_manifest=json.loads((roi_root/'manifest.json').read_text())
    source=ROOT/'tmp/grid_raw'/args.label if args.method=='grid4d_33k' else ROOT/'tmp/baseline_raw'/args.label/args.method
    optimization=args.method in ['grid4d_'+x for x in ['continue','parts','alpha','combined']]
    if optimization:source=ROOT/'optimization/raw'/f'{args.label}_{args.method.removeprefix("grid4d_")}'
    assert (source/'complete.json').exists()
    out=ROOT/'audit/metrics'/args.label;out.mkdir(parents=True,exist_ok=True)
    if optimization:out=ROOT/'optimization/results'/f'{args.label}_{args.method.removeprefix("grid4d_")}'
    path=out/f'{args.method}.csv';assert not path.exists()
    rows=[];streak_summary=[]
    kernel=np.ones((9,9),np.uint8)
    for view in [0,8,16]:
        with np.load(source/f'view{view:02d}.npz') as z:rgbs,alphas=z['rgb'],z['alpha']
        with np.load(roi_root/f'view{view:02d}.npz') as z:labels=z['labels']
        with np.load(roi_root/f'flow{view:02d}.npz') as z:flows,valid=z['backward'],z['valid']
        streak=np.zeros(6,int);longest=np.zeros(6,int)
        prev_pred=prev_target=None
        for frame in range(len(labels)):
            im=np.asarray(Image.open(BASE/'samples'/args.label/'data/images'/f'cam{view:02d}'/f'{frame:06d}.png'),np.float32)/255
            target=im[...,:3]*im[...,3:]+.5*(1-im[...,3:]);fg=im[...,3]>.5
            alpha=alphas[frame];pred=np.clip(rgbs[frame]+.5*(1-alpha[...,None]),0,1)
            error=np.square(pred-target).mean(-1)
            silhouette=alpha>.5
            predicted_edge=silhouette^(cv2.erode(silhouette.astype(np.uint8),np.ones((3,3),np.uint8))>0)
            reference_edge=fg^(cv2.erode(fg.astype(np.uint8),np.ones((3,3),np.uint8))>0)
            edge_distance=cv2.distanceTransform((~predicted_edge).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE) if predicted_edge.any() else np.full((640,352),math.hypot(640,352),np.float32)
            if frame:
                flow=flows[frame-1]
                residual=(pred-remap(prev_pred,flow))-(target-remap(prev_target,flow))
                temporal_error=np.square(residual).mean(-1)
                previous_labels=remap(labels[frame-1],flow,True)
            for p,part in enumerate(PARTS):
                mask=labels[frame]==p+1;area=int(mask.sum())
                row=dict(label=args.label,cohort='reserve' if args.label=='N11' else 'main',method=args.method,view=view,frame=frame,seconds=frame/25,
                    part=part,reference_core_pixels=area,reference_foreground_pixels=int(fg.sum()),
                    unknown_foreground_pixels=int((fg&(labels[frame]==0)).sum()),roi_type='estimated_reference_core',status='measured' if area else 'unknown_or_no_reliable_visible_core',
                    coverage=None,soft_missing=None,recall_alpha05=None,local_window_iou=None,mse=None,psnr=None,boundary_distance_px=None,
                    boundary_reference_pixels=0,outside_reference_alpha=None,outside_window_pixels=0,
                    temporal_mse=None,temporal_valid_pixels=0,temporal_valid_fraction=None)
                if area:
                    coverage=float(alpha[mask].mean())
                    band=cv2.dilate(mask.astype(np.uint8),kernel)>0
                    # Exclude adjacent reference body pixels from this local IoU domain;
                    # retain ALL predicted spill in the reference-defined exterior band.
                    window=band&((~fg)|mask)
                    outside=window&(~fg)
                    edge=band&reference_edge
                    intersection=int((silhouette&mask).sum());union=int(((silhouette&window)|mask).sum())
                    mse=float(error[mask].mean())
                    row.update(coverage=coverage,soft_missing=1-coverage,recall_alpha05=float(silhouette[mask].mean()),local_window_iou=intersection/max(union,1),
                        mse=mse,psnr=-10*math.log10(max(mse,1e-12)),boundary_distance_px=float(edge_distance[edge].mean()) if edge.any() else None,
                        boundary_reference_pixels=int(edge.sum()),outside_reference_alpha=float(alpha[outside].mean()) if outside.any() else None,outside_window_pixels=int(outside.sum()))
                    streak[p]=streak[p]+1 if coverage<.5 else 0;longest[p]=max(longest[p],streak[p])
                    if frame:
                        trusted=mask&valid[frame-1]&(previous_labels==p+1)
                        row.update(temporal_valid_pixels=int(trusted.sum()),temporal_valid_fraction=float(trusted.sum()/area),
                            temporal_mse=float(temporal_error[trusted].mean()) if trusted.any() else None)
                else:
                    streak[p]=0
                rows.append(row)
            prev_pred,prev_target=pred,target
        streak_summary.extend(dict(view=view,part=part,longest_coverage_below05_frames=int(longest[p]),longest_missing_seconds=float(longest[p]/25)) for p,part in enumerate(PARTS))
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(state='complete_reference_core_metrics',label=args.label,method=args.method,rows=len(rows),
        longest_missing=streak_summary,definition=dict(roi='Conservative frozen reference-defined anatomical core; not whole semantic part and not manual GT',
            soft_coverage='mean predicted accumulated alpha on reference core; missing predicted part still evaluated',
            local_iou='Threshold0.5 within9x9-dilated core domain, excluding neighboring reference body pixels',
            temporal='MSE[(pred_now-warp(pred_prev))-(ref_now-warp(ref_prev))] on common reliable reference correspondences of same part',
            temporal_limits='Estimated optical flow; report valid fraction. Missing reference core/correspondence is NA, not zero error',
            old_training_history='Historical sampling traces/counts exist and are audited separately; unrecorded gradients/densification cannot be inferred from terminal assets'))
    (out/f'{args.method}.json').write_text(json.dumps(result,indent=2)+'\n');print(args.label,args.method,'local metrics complete',flush=True)


if __name__=='__main__':main()
