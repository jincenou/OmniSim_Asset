"""Finalize provisional ROI exclusions before any backend-local metrics; reuse unchanged flows."""
import hashlib
import json
import cv2
import numpy as np
from PIL import Image,ImageDraw
from prepare_diagnostics import ROOT,BASE,PARTS,COLORS,project,conservative


def main():
    assert not (ROOT/'audit/part_queue.json').exists()
    q=json.loads((ROOT/'audit/diagnostic_queue.json').read_text());assert q['state']=='complete'
    changes=[]
    for label in ['N01','N02','N03','N04','N06','N07','N08','N11']:
        out=ROOT/'audit/roi'/label;sample=BASE/'samples'/label
        manifest=json.loads((out/'manifest.json').read_text());oldrows={(v['view'],v['frame']):v for v in manifest['rows']}
        points=np.load(sample/'generation/preflight/conditioning/keypoints_3d.npy');n=len(points)
        cams=json.loads((sample/'generation/outputs/turbo/cameras.json').read_text())['cameras'];rows=[]
        contact=Image.new('RGB',(1056,2010),'#333333');draw=ImageDraw.Draw(contact)
        for row,view in enumerate([0,8,16]):
            cam=next(c for c in cams if c['camera_id']==view);K=np.asarray(cam['K'],float);K[:2]*=.5
            path=out/f'view{view:02d}.npz';previous=hashlib.sha256(path.read_bytes()).hexdigest();labels=[]
            for frame in range(n):
                rgba=np.asarray(Image.open(sample/'data/images'/f'cam{view:02d}'/f'{frame:06d}.png'),np.float32)/255;mask=rgba[...,3]>.5
                uv,_=project(points[frame],cam['camera_to_world'],K);roi,_=conservative(uv,mask,label,view);labels.append(roi)
                record=oldrows[view,frame];record.update(core_pixels=[int((roi==p+1).sum()) for p in range(6)],unknown_pixels=int((mask&(roi==0)).sum()));rows.append(record)
                if frame in [0,(n-1)//2,n-1]:
                    col=[0,(n-1)//2,n-1].index(frame);visual=(rgba[...,:3]*rgba[...,3:]+.5*(1-rgba[...,3:]))*255
                    for p,color in enumerate(COLORS):visual[roi==p+1]=.55*visual[roi==p+1]+.45*color
                    contact.paste(Image.fromarray(np.clip(visual,0,255).astype(np.uint8)),(col*352,row*670+30));draw.text((col*352+5,row*670+8),f'{label} v{view:02d} f{frame} CORE/UNKNOWN',fill='white')
            np.savez_compressed(path,labels=np.stack(labels));changes.append(dict(label=label,view=view,previous_provisional_sha256=previous,final_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        contact.save(out/'review.png');manifest['rows']=rows
        manifest['exclusions'][0]='Arms inside torso polygon expanded by0.15*projected shoulder-hip length, to conservatively exclude uncertain occlusion'
        manifest['finalization']='Before per-backend local metrics; based on reference/skeleton overlap review, not selected by method scores. Flow unchanged.'
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/'audit/roi_protocol_frozen.json').write_text(json.dumps(dict(state='frozen_before_part_metrics',changes=changes,
        part_roi_scope='Estimated anatomical cores only. Full body/hair/accessory/occluded-part coverage is not claimed. Unknown coverage is reported.',
        note='No old Random8/CEM result changed. Only unpublished ROI proposals finalized; provisional hashes retained.'),indent=2)+'\n')
    print('All8 reference core definitions frozen; flows unchanged',flush=True)


if __name__=='__main__':main()
