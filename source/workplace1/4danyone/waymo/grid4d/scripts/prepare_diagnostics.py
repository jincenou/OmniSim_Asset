"""Conservative reviewed reference cores and reliability-gated motion correspondences."""
import argparse
import json
import cv2
import numpy as np
from PIL import Image,ImageDraw
from roi_candidates import ROOT,BASE,PARTS,COLORS,project,segment,candidates,XX,YY


def remap(image,flow,nearest=False):
    return cv2.remap(image,XX+flow[...,0],YY+flow[...,1],
        cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)


def conservative(uv,mask,label,view):
    roi,support=candidates(uv,mask)
    torso=max(float((np.linalg.norm(uv[5]-uv[9])+np.linalg.norm(uv[6]-uv[10]))/2),10.)
    polygon=np.zeros_like(mask,np.uint8)
    cv2.fillConvexPoly(polygon,np.rint(uv[[5,6,10,9]]).astype(np.int32),1)
    arm_dist=np.minimum.reduce([segment(uv[a],uv[b]) for a,b in [(5,7),(7,62),(6,8),(8,41)]])
    # Occlusion cannot be resolved reliably from a projected skeleton alone.
    radius=max(1,int(round(torso*.15)))
    expanded=cv2.dilate(polygon,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))>0
    roi[((roi==3)|(roi==4))&expanded]=0
    roi[(roi==2)&((polygon==0)|(arm_dist<torso*.115*1.5))]=0
    left=np.minimum(segment(uv[9],uv[11]),segment(uv[11],uv[13]))
    right=np.minimum(segment(uv[10],uv[12]),segment(uv[12],uv[14]))
    roi[((roi==5)|(roi==6))&(left<torso*.145*1.5)&(right<torso*.145*1.5)]=0
    if label=='N02':
        # Recorded review: bag/sheet-like generated material prevents reliable anatomical attribution.
        roi[(roi==5)|(roi==6)]=0
    if label=='N11':
        for part,a,b in [(5,9,11),(6,10,12)]:
            upper=segment(uv[a],uv[a]+.55*(uv[b]-uv[a]))<torso*.145*1.5
            roi[(roi==part)&upper]=0
    for part in range(1,7):
        if int((roi==part).sum())<64:roi[roi==part]=0
    return roi,support


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);args=parser.parse_args()
    cv2.setNumThreads(2)
    sample=BASE/'samples'/args.label;out=ROOT/'audit/roi'/args.label
    out.mkdir(parents=True,exist_ok=False)
    points=np.load(sample/'generation/preflight/conditioning/keypoints_3d.npy')
    cameras=json.loads((sample/'generation/outputs/turbo/cameras.json').read_text())['cameras']
    n=len(points);stats=[];flowstats=[]
    contact=Image.new('RGB',(1056,2010),'#333333');draw=ImageDraw.Draw(contact)
    for row,view in enumerate([0,8,16]):
        cam=next(c for c in cameras if c['camera_id']==view);K=np.asarray(cam['K'],float);K[:2]*=.5
        labels=[];flows=[];valids=[]
        prev_gray=prev_mask=prev_rgb=None
        for frame in range(n):
            rgba=np.asarray(Image.open(sample/'data/images'/f'cam{view:02d}'/f'{frame:06d}.png'),np.float32)/255
            rgb=rgba[...,:3]*rgba[...,3:]+.5*(1-rgba[...,3:]);mask=rgba[...,3]>.5
            uv,_=project(points[frame],cam['camera_to_world'],K)
            roi,support=conservative(uv,mask,args.label,view);labels.append(roi)
            stats.append(dict(view=view,frame=frame,seconds=frame/25,reference_pixels=int(mask.sum()),
                core_pixels=[int((roi==p+1).sum()) for p in range(6)],unknown_pixels=int((mask&(roi==0)).sum()),
                centerline_support=support,roi_status='estimated_reference_core_not_semantic_ground_truth'))
            gray=cv2.cvtColor(np.rint(rgb*255).astype(np.uint8),cv2.COLOR_RGB2GRAY)
            if prev_gray is not None:
                backward=cv2.calcOpticalFlowFarneback(gray,prev_gray,None,.5,3,15,3,5,1.2,0)
                forward=cv2.calcOpticalFlowFarneback(prev_gray,gray,None,.5,3,15,3,5,1.2,0)
                cycle=backward+remap(forward,backward)
                cycle_error=np.linalg.norm(cycle,axis=-1)
                mapx,mapy=XX+backward[...,0],YY+backward[...,1]
                photo=np.mean(np.abs(rgb-remap(prev_rgb,backward)),axis=-1)
                valid=mask&(remap(prev_mask.astype(np.uint8),backward,True)>0)&(mapx>=0)&(mapx<=351)&(mapy>=0)&(mapy<=639)
                valid&=(cycle_error<=1.+.05*np.linalg.norm(backward,axis=-1))&(photo<=.10)
                flows.append(backward);valids.append(valid)
                flowstats.append(dict(view=view,frame=frame,reference_pixels=int(mask.sum()),valid_pixels=int(valid.sum()),
                    valid_fraction=float(valid.sum()/max(mask.sum(),1)),mean_cycle_error_valid=float(cycle_error[valid].mean()) if valid.any() else None))
            prev_gray,prev_mask,prev_rgb=gray,mask,rgb
            if frame in [0,(n-1)//2,n-1]:
                col=[0,(n-1)//2,n-1].index(frame);visual=rgb*255
                for p,color in enumerate(COLORS):visual[roi==p+1]=.55*visual[roi==p+1]+.45*color
                contact.paste(Image.fromarray(np.clip(visual,0,255).astype(np.uint8)),(col*352,row*670+30))
                draw.text((col*352+5,row*670+8),f'{args.label} v{view:02d} f{frame} CORE/UNKNOWN',(255,255,255))
        np.savez_compressed(out/f'view{view:02d}.npz',labels=np.stack(labels))
        np.savez_compressed(out/f'flow{view:02d}.npz',backward=np.stack(flows),valid=np.stack(valids))
    contact.save(out/'review.png')
    (out/'manifest.json').write_text(json.dumps(dict(label=args.label,parts=PARTS,rows=stats,flow_rows=flowstats,
        roi_definition='Reference RGBA intersection with conservative projected anatomical cores; ambiguous and accessory regions remain unknown, not zero-error',
        manual_review='Fixed first/middle/last of all3 H views reviewed at candidate stage; conservative overlap exclusion applied before per-method metric calculation',
        exclusions=['Arms projected inside torso polygon','Torso intersecting arm capsules','Overlapping leg capsules','N02 leg/accessory semantics uncertain','N11 upper-thigh dress region','Core smaller than64pixels'],
        flow=dict(algorithm='OpenCV Farneback',pyr_scale=.5,levels=3,winsize=15,iterations=3,poly_n=5,poly_sigma=1.2,
            backward_direction='current pixel to previous pixel',cycle_threshold='1px+0.05*motion magnitude',photometric_mean_abs_threshold=.10,
            validity='Both reference masks,inside image,forward/backward consistency,reference photometric agreement',
            limitation='Estimated correspondence, not ground-truth optical flow; uncovered regions remain unmeasured')),
        indent=2)+'\n')
    print(args.label,'conservative core/flow preparation complete',flush=True)


if __name__=='__main__':main()
