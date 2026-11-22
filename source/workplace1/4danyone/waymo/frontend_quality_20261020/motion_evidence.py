"""Identity-independent evidence for motion completion and camera separation.

This module does not assert hidden poses are ground truth, fill a gap by forcing
walking, or treat image displacement as world motion.
"""
import numpy as np
from scipy.spatial.transform import Rotation,Slerp

COCO_NAMES=('nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder',
 'left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle')

def uncertain_intervals(keypoints,frame_times,confidence=.3,min_duration=.12):
 k=np.asarray(keypoints);t=np.asarray(frame_times,dtype=float)
 if k.ndim!=3 or k.shape[1:]!=(17,3) or len(t)!=len(k) or not np.isfinite(k).all() or not np.all(np.diff(t)>0):
  raise ValueError('Expected finite [time,17,xyconfidence] and increasing times')
 intervals=[];dt=float(np.median(np.diff(t)))
 for joint in range(5,17):
  bad=k[:,joint,2]<confidence
  edges=np.diff(np.r_[False,bad,False].astype(np.int8));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
  for start,end in zip(starts,ends,strict=True):
   duration=float(t[end-1]-t[start]+dt)
   if duration+1e-9<min_duration:continue
   intervals.append(dict(joint=COCO_NAMES[joint],start_slot=int(start),end_slot_inclusive=int(end-1),duration_seconds=duration,
    bounded_by_observations=bool(start>0 and end<len(t)),reason='low_detector_confidence_not_occlusion_ground_truth'))
 return dict(requires_completion_review=bool(intervals),intervals=intervals,threshold=confidence,min_duration_seconds=min_duration,
  scope='Low confidence is missing-evidence candidate; high confidence does not establish visibility or correct pose.',quality_approved=False)

def resample_measured_camera(native_times,camera_to_world,query_times):
 t=np.asarray(native_times,dtype=float);q=np.asarray(query_times,dtype=float);c=np.asarray(camera_to_world,dtype=float)
 if c.shape!=(len(t),4,4) or not np.isfinite(c).all() or not np.all(np.diff(t)>0) or not np.isfinite(q).all():raise ValueError('Invalid measured cameras/times')
 if not np.allclose(c[:,3,:],[0,0,0,1],atol=1e-8):raise ValueError('Invalid homogeneous row')
 if not np.allclose(c[:,:3,:3]@c[:,:3,:3].transpose(0,2,1),np.eye(3),atol=1e-6) or not np.allclose(np.linalg.det(c[:,:3,:3]),1,atol=1e-6):raise ValueError('Rotation is not SO3')
 # Relative times avoid loss of precision inside interpolation; no extrapolation.
 if q.min()<t[0] or q.max()>t[-1]:raise ValueError('Query outside measured time support')
 result=np.repeat(np.eye(4)[None],len(q),axis=0)
 result[:,:3,:3]=Slerp(t-t[0],Rotation.from_matrix(c[:,:3,:3]))(q-t[0]).as_matrix()
 result[:,:3,3]=np.column_stack([np.interp(q-t[0],t-t[0],c[:,i,3]) for i in range(3)])
 return result

def points_to_world(points_incam,camera_to_world):
 p=np.asarray(points_incam,dtype=float);c=np.asarray(camera_to_world,dtype=float)
 if p.ndim!=3 or p.shape[-1]!=3 or c.shape!=(len(p),4,4) or not np.isfinite(p).all():raise ValueError('Invalid camera-space points')
 return np.einsum('tij,tkj->tki',c[:,:3,:3],p)+c[:,None,:3,3]
