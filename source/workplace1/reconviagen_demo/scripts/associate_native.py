"""Audit projected 3D track correspondence, without equating 2D and 3D IDs."""
from native_candidates import ROOT, frames
import json
import numpy as np
from pathlib import Path

def xyxy(b):return np.array([b.center_x-b.length/2,b.center_y-b.width/2,b.center_x+b.length/2,b.center_y+b.width/2])
def iou(a,b):
    v=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])); inter=v.prod()
    return float(inter/(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter+1e-9))

for name in ['service_van','silver_sedan_03','white_sedan']:
    p=ROOT/'inputs'/name/'native_inventory.json';d=json.loads(p.read_text())
    rows={}
    for r in d['observations']:rows.setdefault(r['frame_index'],[]).append(r)
    for fi,frame in frames(Path(d['raw_tfrecord'])):
        ls={l.id:l for l in frame.laser_labels};proj={c.name:c.labels for c in frame.projected_lidar_labels}
        for r in rows.get(fi,[]):
            hits=[]
            for l in proj.get(r['camera'],[]):
                ids=[lid for lid in ls if l.id.startswith(lid+'_')]
                if len(ids)!=1:continue
                hits.append((iou(np.array(r['box_xyxy']),xyxy(l.box)),ids[0]))
            hits.sort(reverse=True)
            if hits:
                r['projected_best_iou']=hits[0][0];r['projected_iou_margin']=hits[0][0]-(hits[1][0] if len(hits)>1 else 0)
                if hits[0][0]>.4 and r['projected_iou_margin']>.15:
                    r['projected_laser_object_id']=hits[0][1]
                    l=ls[hits[0][1]];b=l.box
                    r['projected_laser_box']=[b.center_x,b.center_y,b.center_z,b.length,b.width,b.height,b.heading]
                    r['projected_speed_mps']=[l.metadata.speed_x,l.metadata.speed_y]
    best=d['candidates'][0];track=[r for r in d['observations'] if r['camera']==best['camera'] and r['camera_2d_id']==best['camera_2d_id']]
    counts={}
    for r in track:
        lid=r.get('projected_laser_object_id')
        if lid:counts[lid]=counts.get(lid,0)+1
    d['projection_audit']=dict(seed_camera=best['camera'],seed_2d_id=best['camera_2d_id'],track_observations=len(track),projected_3d_votes=counts)
    (p.parent/'associated_inventory.json').write_text(json.dumps(d,indent=2))
    print(name,d['projection_audit'],flush=True)
