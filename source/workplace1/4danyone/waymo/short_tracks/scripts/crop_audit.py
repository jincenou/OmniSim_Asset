"""Measure actual conditioning crop retention and projected body coverage."""
import json
import os
from pathlib import Path
import numpy as np

ROOT=Path(os.environ['FDANYONE_WAYMO_ROOT']).resolve()


def main():
    d=ROOT/'preflight/conditioning';meta=json.loads((d/'metadata.json').read_text());crop=meta['source_crop']
    masks=np.load(ROOT/'preflight/source_masks.npz')['masks']>.05*255
    retained=masks[:,crop['top']:crop['top']+crop['height'],crop['left']:crop['left']+crop['width']].sum((1,2))/np.maximum(masks.sum((1,2)),1)
    points=np.load(d/'keypoints_3d.npy')[:,:21]
    camera=json.loads((d/'cameras.json').read_text())['cameras'];views=[]
    for c in camera:
        w=np.array(c['world_to_camera']);p=points@w[:3,:3].T+w[:3,3];q=p@np.array(c['K']).T
        uv=q[...,:2]/q[...,2:];ok=(p[...,2]>0)&(uv[...,0]>=0)&(uv[...,0]<c['image_width'])&(uv[...,1]>=0)&(uv[...,1]<c['image_height'])
        views.append({'view':c['camera_id'],'fully_visible_frames':int(ok.all(1).sum()),'min_body_keypoint_fraction':float(ok.mean(1).min()),'mean_body_keypoint_fraction':float(ok.mean())})
    result={'source_crop':crop,'source_mask_retained_min':float(retained.min()),'source_mask_retained_mean':float(retained.mean()),
            'source_frames_under_95_percent':int((retained<.95).sum()),'target_body_coverage':views,
            'gate_passed':bool(retained.min()>=.95 and min(x['min_body_keypoint_fraction'] for x in views)>=.95),
            'scope':'Estimated foreground and projected conditioning geometry; not measured generated image fidelity'}
    (ROOT/'preflight/crop_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
