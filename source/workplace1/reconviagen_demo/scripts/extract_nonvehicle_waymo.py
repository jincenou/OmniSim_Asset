from pathlib import Path
import json,io,hashlib,sys
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageOps
from native_candidates import frames
R=Path(__file__).resolve().parents[1];O=R/'nonvehicle10';cv2.setNumThreads(2)
cs=json.loads((O/'inputs/candidates.json').read_text());chosen=[0,47,31,29,13,7,2,55];names=['green_light_urban','red_light_suburban','stop_sign','speed30_sign','cylindrical_bin','square_bin','white_hydrant','yellow_hydrant']
for i,(idx,name) in enumerate(zip(chosen,names)):
 r=cs[idx];caseid=f'nonvehicle_{i+1:02}_{name}';out=O/'inputs'/caseid;out.mkdir(exist_ok=True);fi=r['frame_index'];wanted=[fi,fi+1,fi+2,fi+3];raw=Path(r['raw_tfrecord']);ims={};metas={}
 for j,f in frames(raw):
  if j in wanted:
   data=next(x for x in f.images if x.name==r['camera']);im=np.array(Image.open(io.BytesIO(data.image)).convert('RGB'));ims[j]=im;metas[j]=dict(image_timestamp_s=data.pose_timestamp,ego_to_world=list(f.pose.transform),camera_pose_to_world=list(data.pose.transform))
  if j>=wanted[-1]:break
 original=ims[fi];gray=cv2.cvtColor(original,cv2.COLOR_RGB2GRAY);box=np.array(r['box_xyxy']);mask=np.zeros(gray.shape,np.uint8);x0,y0,x1,y1=box.astype(int);mask[y0:y1,x0:x1]=255;pts=cv2.goodFeaturesToTrack(gray,150,.005,3,mask=mask);rows=[]
 if pts is None or len(pts)<6:raise ValueError((caseid,'insufficient tracking points'))
 for k,j in enumerate(wanted):
  im=ims[j];bbox=box.copy();audit={'method':'source detection','inliers':len(pts)}
  if j!=fi:
   g=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY);p1,st,err=cv2.calcOpticalFlowPyrLK(gray,g,pts,None,winSize=(31,31),maxLevel=4);back,stb,errb=cv2.calcOpticalFlowPyrLK(g,gray,p1,None,winSize=(31,31),maxLevel=4);good=(st[:,0]>0)&(stb[:,0]>0)&(np.linalg.norm(back[:,0]-pts[:,0],axis=1)<1.5);a=pts[good,0];b=p1[good,0]
   if len(a)<6:
    print(caseid,j,'SKIP unreliable association',len(a),flush=True);continue
   M,ins=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2.5)
   if M is None or ins.sum()<6:
    print(caseid,j,'SKIP unreliable affine',flush=True);continue
   corners=np.array([[box[0],box[1],1],[box[2],box[1],1],[box[2],box[3],1],[box[0],box[3],1]])@M.T;bbox=np.r_[corners.min(0),corners.max(0)];audit=dict(method='same-camera forward/backward KLT + RANSAC affine',inliers=int(ins.sum()),tracked_points=int(good.sum()),affine=M.tolist(),median_residual_px=float(np.median(np.linalg.norm(np.c_[a,np.ones(len(a))]@M.T-b,axis=1))))
  pad=.12*max(bbox[2]-bbox[0],bbox[3]-bbox[1]);crop=[max(0,int(bbox[0]-pad)),max(0,int(bbox[1]-pad)),min(im.shape[1],int(bbox[2]+pad)),min(im.shape[0],int(bbox[3]+pad))];path=out/f'real_{k:02}.png';Image.fromarray(im[crop[1]:crop[3],crop[0]:crop[2]]).save(path)
  rows.append(dict(r,**metas[j],id=f'real_{k:02}',source_type='real',split='train',frame_index=j,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),box_xyxy=bbox.tolist(),crop_box=crop,box_crop_xyxy=(bbox-np.array([crop[0],crop[1],crop[0],crop[1]])).tolist(),association=audit))
 Image.fromarray(original).save(out/'source_full.jpg',quality=92)
 case=dict(case_id=caseid,category=r['category'],dataset='waymo',original_obj_id=r['candidate_id'],flux_reference_id='real_00',images=rows,selection='Source verified before generation; up to four distinct nearby observations; rejected unreliable associations are omitted; no heldout subset',matched_source_pose_available=False,scale_source='unknown; normalized-only comparison',setup='GroundingDINO detection + verified KLT short association + SAM2; no object 3D pose or size used')
 (out/'manifest.json').write_text(json.dumps(case,indent=2));print(caseid,'DONE',flush=True)
