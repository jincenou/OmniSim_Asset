"""Audit real saved sampling ledgers without retraining historical assets."""
import argparse
import csv
import json
import numpy as np
from PIL import Image
import cv2
from prepare_diagnostics import ROOT,BASE,PARTS,conservative,project
from render_baseline import sha,save


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);args=parser.parse_args();label=args.label
    cv2.setNumThreads(2);sample=BASE/'samples'/label;data=json.loads((BASE/'four_dg/data'/label/'fdanyone.json').read_text());frames=data['frame_count']
    out=ROOT/'audit/historical_exposure'/label;out.mkdir(parents=True,exist_ok=False)
    areas=np.zeros((24,frames,6),np.int64);prior=ROOT/'audit/training'/label/'exposure.csv'
    if prior.exists():
        rows=list(csv.DictReader(prior.open()));assert len(rows)==21*frames*6
        for row in rows:areas[int(row['view']),int(row['frame']),PARTS.index(row['part'])]=int(row['reference_core_pixels'])
        origin=dict(path=str(prior),sha256=sha(prior))
    else:
        points=np.load(sample/'generation/preflight/conditioning/keypoints_3d.npy')
        cams=json.loads((sample/'generation/outputs/turbo/cameras.json').read_text())['cameras']
        for view in data['train_views']:
            c=next(c for c in cams if c['camera_id']==view);k=np.asarray(c['K'],float);k[:2]*=.5
            for frame in range(frames):
                mask=np.asarray(Image.open(sample/'data/images'/f'cam{view:02d}'/f'{frame:06d}.png'))[...,3]>127
                uv,_=project(points[frame],c['camera_to_world'],k);roi,_=conservative(uv,mask,label,view)
                areas[view,frame]=np.bincount(roi.flatten(),minlength=7)[1:]
        origin=dict(core_definition_sha256=sha(ROOT/'scripts/prepare_diagnostics.py'),source='Frozen train-only RGBA and conditioning projections; not per-method predictions')
    assert not areas[[0,8,16]].any();np.save(out/'core_areas.npy',areas)
    rows=[];proofs={};ledgers={}
    for method in ['ftgs_base5k','ftgs_opt15k','4dgs_grid25','4dgs_grid61','4dgs_grid61_snapshot12000']:
        if method.startswith('ftgs'):
            path=sample/method/'training_complete.json';done=json.loads(path.read_text())
            counts=np.asarray(done['sampling']['counts_by_camera_and_frame'],np.int64);updates=counts.copy()
            assert updates.sum()==done['actual_optimizer_updates'];source=dict(path=str(path),sha256=sha(path),scope=done['sampling']['count_scope'])
        else:
            run=BASE/'four_dg/runs'/f'{label}_grid{25 if method.endswith("25") else 61}'
            path=run/'sampling_trace.npy';trace=np.load(path);done=json.loads((run/'complete.json').read_text())
            assert sha(path)==done['sampling_trace_sha256'];assert trace.shape==(33000,3)
            sampling=json.loads((run/'point_cloud/iteration_30000/sampling.json').read_text())
            for stage,name in [(0,'coarse'),(1,'fine')]:
                actual=np.zeros((24,frames),np.int64);t=trace[trace[:,0]==stage];np.add.at(actual,(t[:,1],t[:,2]),1)
                assert np.array_equal(actual,np.asarray(sampling[name]))
            chosen=trace[:15000] if method.endswith('snapshot12000') else trace
            counts=np.zeros((24,frames),np.int64);np.add.at(counts,(chosen[:,1],chosen[:,2]),1)
            updates=counts.copy();updates[int(chosen[-1,1]),int(chosen[-1,2])]-=1
            assert updates.sum()==len(chosen)-1
            source=dict(path=str(path),sha256=sha(path),scope='Actual sampled renders; final sample has no optimizer update in this saved asset',snapshot_before_optimizer_step=method.endswith('snapshot12000'))
        assert counts.shape==(24,frames) and not counts[[0,8,16]].any() and not updates[[0,8,16]].any()
        ledgers[method+'_renders']=counts;ledgers[method+'_updates']=updates
        proofs[method]=dict(**source,renders=int(counts.sum()),actual_updates=int(updates.sum()))
        for i,part in enumerate(PARTS):
            a=areas[...,i];valid=a>0;pixels=int(a.sum())
            rows.append(dict(label=label,cohort='reserve' if label=='N11' else 'main',method=method,part=part,valid_pairs=int(valid.sum()),reference_pixels=pixels,
                actual_updates=int(updates.sum()),pixel_renders=int((a*counts).sum()),pixel_updates=int((a*updates).sum()),
                mean_updates_per_reference_pixel=float((a*updates).sum()/pixels) if pixels else None,
                min_pair_updates=int(updates[valid].min()) if valid.any() else None,max_pair_updates=int(updates[valid].max()) if valid.any() else None))
    np.savez_compressed(out/'ledgers.npz',**ledgers)
    with (out/'summary.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(out/'complete.json',dict(state='passed',label=label,core_source=origin,sources=proofs,heldout_content_read=False,
        correction='Historical32 sampling/exposure is recorded. Historical comparable per-step gradients and densification events remain not_recorded; terminal support cannot recover them.',
        semantics='Area-weighted exposure of conservative training reference cores; not semantic ground truth nor observed screen gradients'))
    print(label,'sampling ledgers and train-only part exposure audited',flush=True)


if __name__=='__main__':main()
