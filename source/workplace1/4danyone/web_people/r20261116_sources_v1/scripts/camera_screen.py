"""Background-only feature motion diagnostic; never a calibration substitute."""
import sys,json
import av,cv2,numpy as np
from prepare_person import ROOT,selected,save
for aid in sys.argv[1:]:
 choice=selected(aid);d=ROOT/'inputs'/aid;mapping=json.loads((d/'inputs/frame_mapping.json').read_text())['frames']
 wanted={r['source_frame_index']:r for r in mapping};grays=[];boxes=[]
 with av.open(choice['source_path']) as c:
  c.streams.video[0].thread_count=2
  for i,f in enumerate(c.decode(video=0)):
   if i in wanted:
    rgb=f.to_ndarray(format='rgb24');scale=960/rgb.shape[1]
    gray=cv2.cvtColor(cv2.resize(rgb,(960,round(rgb.shape[0]*scale))),cv2.COLOR_RGB2GRAY);grays.append(gray)
    boxes.append(np.asarray(wanted[i]['bbox_source_xyxy'])*scale)
   if i>=max(wanted):break
 mask=np.full(grays[0].shape,255,np.uint8)
 # Exclude the entire target trajectory to avoid measuring person movement.
 for box in boxes:
  x0,y0,x1,y1=box;pad=30
  mask[max(0,int(y0-pad)):int(y1+pad),max(0,int(x0-pad)):int(x1+pad)]=0
 corners=cv2.goodFeaturesToTrack(grays[0],maxCorners=500,qualityLevel=.01,minDistance=10,mask=mask)
 records=[]
 if corners is not None and len(corners)>=20:
  for frame,gray in enumerate(grays):
   pts,status,error=cv2.calcOpticalFlowPyrLK(grays[0],gray,corners,None,winSize=(31,31),maxLevel=4)
   good=(status[:,0]>0)&(error[:,0]<30);a=corners[:,0][good];b=pts[:,0][good]
   if len(a)<12:records.append(dict(frame=frame,valid=False));continue
   matrix,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2)
   if matrix is None:records.append(dict(frame=frame,valid=False));continue
   ids=inliers[:,0].astype(bool);motion=np.linalg.norm(b[ids]-a[ids],axis=1)
   records.append(dict(frame=frame,valid=True,matrix_960px=matrix.tolist(),inliers=int(ids.sum()),median_background_displacement_px=float(np.median(motion)),p95_background_displacement_px=float(np.quantile(motion,.95))))
 save(d/'research/background_camera_screen.json',dict(method='LK original background; target trajectory excluded; first-frame reference at width960',
  source_is_original=True,frames=records,valid_frames=sum(r['valid'] for r in records),
  calibration=False,automatic_acceptance=False,limitations='Sparse background motion only; cannot distinguish rotation/translation/zoom or recover metric camera'))
 valid=[r['median_background_displacement_px'] for r in records if r['valid']]
 print(aid,len(valid),'max median displacement',max(valid) if valid else None,flush=True)
