"""Reference-defined anatomical ROI candidates, explicitly not semantic truth."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent/'random8/r20261015_113332'
PARTS=['head','torso','left_arm','right_arm','left_leg','right_leg']
COLORS=np.array([[255,160,30],[60,190,240],[235,65,150],[125,75,235],[70,210,80],[230,215,40]],np.float32)
YY,XX=np.mgrid[:640,:352].astype(np.float32)


def project(points,pose,K):
    w=np.linalg.inv(np.asarray(pose,dtype=np.float64));p=points@w[:3,:3].T+w[:3,3]
    q=p@np.asarray(K).T
    return q[:,:2]/q[:,2:],p[:,2]


def segment(a,b):
    delta=b-a
    u=np.clip(((XX-a[0])*delta[0]+(YY-a[1])*delta[1])/max(float(delta@delta),1e-6),0,1)
    return np.sqrt((XX-a[0]-u*delta[0])**2+(YY-a[1]-u*delta[1])**2)


def candidates(uv,mask):
    torso=max(float((np.linalg.norm(uv[5]-uv[9])+np.linalg.norm(uv[6]-uv[10]))/2),10.)
    skeleton=[[(69,0)],[(5,9),(6,10),(5,6),(9,10)],[(5,7),(7,62)],[(6,8),(8,41)],[(9,11),(11,13),(13,15)],[(10,12),(12,14),(14,18)]]
    radii=np.asarray([.22,.28,.115,.115,.145,.145])*torso
    scores=[];supports=[]
    for part,bones in enumerate(skeleton):
        dist=np.minimum.reduce([segment(uv[a],uv[b]) for a,b in bones])/radii[part]
        if part==1:
            poly=np.zeros_like(mask,np.uint8);cv2.fillConvexPoly(poly,np.rint(uv[[5,6,10,9]]).astype(np.int32),1)
            dist[poly>0]=0
        scores.append(dist)
        line=np.concatenate([np.linspace(uv[a],uv[b],20) for a,b in bones])
        x=np.clip(np.rint(line[:,0]).astype(int),0,351);y=np.clip(np.rint(line[:,1]).astype(int),0,639)
        valid=(line[:,0]>=0)&(line[:,0]<352)&(line[:,1]>=0)&(line[:,1]<640)
        supports.append(float((valid&mask[y,x]).mean()))
    distances=np.stack(scores)
    order=np.argsort(distances,axis=0)
    first=np.take_along_axis(distances,order[:1],axis=0)[0]
    second=np.take_along_axis(distances,order[1:2],axis=0)[0]
    labels=(order[0]+1).astype(np.uint8)
    # Unassigned includes uncertain overlaps, not discarded prediction failures.
    labels[(first>1.5)|((second-first)<.25)|(~mask)]=0
    for p,support in enumerate(supports):
        if support<.65:labels[labels==p+1]=0
    return labels,supports


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);args=parser.parse_args()
    sample=BASE/'samples'/args.label;out=ROOT/'audit/roi_candidates'/args.label
    out.mkdir(parents=True,exist_ok=False)
    pts_path=sample/'generation/preflight/conditioning/keypoints_3d.npy'
    points=np.load(pts_path)
    cams=json.loads((sample/'generation/outputs/turbo/cameras.json').read_text())['cameras']
    meta=json.loads((BASE/'four_dg/data'/args.label/'fdanyone.json').read_text())
    mapping=np.asarray(meta['source_to_normalized'])
    n=len(points);assert n==meta['frame_count']
    stats=[];contact=Image.new('RGB',(352*3,670*3),'#333333');draw=ImageDraw.Draw(contact)
    for row,view in enumerate([0,8,16]):
        record=next(c for c in cams if c['camera_id']==view)
        K=np.asarray(record['K'],float);K[:2]*=.5
        normalized_cam=next(c for c in meta['cameras'] if c['view']==view)
        labels=[]
        for frame in range(n):
            uv,depth=project(points[frame],record['camera_to_world'],K)
            normalized=points[frame]@mapping[:3,:3].T+mapping[:3,3]
            second,_=project(normalized,normalized_cam['camtoworld'],normalized_cam['K'])
            assert np.abs(uv-second).max()<.001
            rgba=np.asarray(Image.open(sample/'data/images'/f'cam{view:02d}'/f'{frame:06d}.png'),np.float32)/255
            mask=rgba[...,3]>.5
            roi,support=candidates(uv,mask)
            labels.append(roi)
            stats.append(dict(view=view,frame=frame,seconds=frame/25,reference_foreground_pixels=int(mask.sum()),
                assigned_pixels=[int((roi==p+1).sum()) for p in range(6)],centerline_reference_support=support,
                unknown_reference_pixels=int((mask&(roi==0)).sum()),status='candidate_requires_visual_review'))
            if frame in [0,(n-1)//2,n-1]:
                col=[0,(n-1)//2,n-1].index(frame)
                rgb=rgba[...,:3]*rgba[...,3:]+.5*(1-rgba[...,3:])
                visual=rgb*255
                for p,color in enumerate(COLORS):visual[roi==p+1]=.55*visual[roi==p+1]+.45*color
                image=Image.fromarray(np.clip(visual,0,255).astype(np.uint8))
                ink=ImageDraw.Draw(image)
                for x,y in uv[[0,5,6,7,8,9,10,11,12,13,14,41,62,69]]:
                    ink.ellipse((x-2,y-2,x+2,y+2),fill='white')
                contact.paste(image,(col*352,row*670+30))
                draw.text((col*352+6,row*670+8),f'{args.label} view{view:02d} frame{frame} CANDIDATE',(255,255,255))
        np.savez_compressed(out/f'view{view:02d}.npz',labels=np.stack(labels))
    contact.save(out/'review.png')
    result=dict(label=args.label,state='candidate_only_not_semantic_truth',parts=PARTS,
        thresholds=dict(max_normalized_distance=1.5,min_assignment_margin=.25,min_centerline_support=.65),
        source=str(pts_path),source_sha256=hashlib.sha256(pts_path.read_bytes()).hexdigest(),
        reference='Frozen generated RGBA alpha>0.5; no prediction-based ROI; uncertain overlap and low alignment remain unknown',
        limitations='Conditioning skeleton can disagree with generated appearance; centerline support is not a calibrated confidence or proof of part visibility; inspect overlays before metrics',
        rows=stats)
    (out/'candidates.json').write_text(json.dumps(result,indent=2)+'\n')
    print(args.label,'candidate ROIs written; review pending',flush=True)


if __name__=='__main__':main()
