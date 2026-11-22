"""Source-only background image-motion diagnostic; never consumed by GVHMR."""
import argparse,json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageDraw
from source_io import read_source,source_identity
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('source_id');p.add_argument('source');p.add_argument('--start',type=int,required=True);p.add_argument('--end',type=int,required=True);a=p.parse_args()
    path=Path(a.source);cv2.setNumThreads(2)
    rows=[];first=None;points=None;contacts=[]
    for index,t,rgb in read_source(path):
        if index<a.start:continue
        if index>a.end:break
        h,w=rgb.shape[:2];gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        if first is None:
            first=gray;mask=np.zeros_like(gray);mask[:int(.35*h),:int(.3*w)]=255;mask[:int(.35*h),int(.8*w):]=255
            points=cv2.goodFeaturesToTrack(first,maxCorners=160,qualityLevel=.02,minDistance=15,mask=mask)
            assert points is not None and len(points)>=30,'Insufficient background corners; camera remains unknown'
        target,valid,_=cv2.calcOpticalFlowPyrLK(first,gray,points,None,winSize=(25,25),maxLevel=4)
        back,valid_back,_=cv2.calcOpticalFlowPyrLK(gray,first,target,None,winSize=(25,25),maxLevel=4)
        good=(valid[:,0]>0)&(valid_back[:,0]>0)&(np.linalg.norm(back[:,0]-points[:,0],axis=1)<.5)
        delta=target[:,0]-points[:,0]
        good &= np.isfinite(delta).all(1)
        values=np.linalg.norm(delta[good],axis=1)
        if len(values)<30:
            rows.append(dict(frame=index,timestamp=t,valid_corners=int(good.sum()),median_displacement_px=None,p90_displacement_px=None))
        else:
            rows.append(dict(frame=index,timestamp=t,valid_corners=int(good.sum()),median_displacement_px=float(np.median(values)),p90_displacement_px=float(np.quantile(values,.9))))
        if index in np.rint(np.linspace(a.start,a.end,5)).astype(int):
            im=Image.fromarray(rgb);d=ImageDraw.Draw(im)
            for q,r in zip(points[:,0][good],target[:,0][good],strict=True):d.line([tuple(q),tuple(r)],fill='yellow',width=2);d.ellipse((r[0]-3,r[1]-3,r[0]+3,r[1]+3),fill='lime')
            im=im.resize((960,int(round(h*960/w/2)*2)));ImageDraw.Draw(im).text((8,8),f'{a.source_id} ORIGINAL frame {index}; diagnostic only',fill='white',stroke_width=2,stroke_fill='black');contacts.append(im)
    assert len(rows)==a.end-a.start+1
    complete=all(r['median_displacement_px'] is not None for r in rows)
    values=[r['median_displacement_px'] for r in rows if r['median_displacement_px'] is not None]
    data=dict(source_identity=source_identity(path),frame_interval=[a.start,a.end],frames_checked=len(rows),
        source_size=[w,h],camera_class='unknown_pending_original_image_review',diagnostic_only=True,
        feeds_model=False,estimates_camera_pose=False,initial_background_corners=len(points),
        all_frames_sufficient_correspondences=complete,maximum_median_displacement_px=max(values) if values else None,
        maximum_median_displacement_over_diagonal=max(values)/np.hypot(w,h) if values else None,
        evidence_limit='Image correspondences do not prove 3D stationarity; residual scene motion and tracking errors possible',rows=rows)
    out=ROOT/'sources'/f'{a.source_id}_camera_{a.start}_{a.end}'
    out.with_suffix('.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    contact=Image.new('RGB',(960,sum(im.height for im in contacts)));y=0
    for im in contacts:contact.paste(im,(0,y));y+=im.height
    contact.save(out.with_suffix('.jpg'),quality=92)
    print(a.source_id,{k:v for k,v in data.items() if k!='rows'})

if __name__=='__main__':main()
