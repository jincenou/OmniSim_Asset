"""Freeze source cameras from native metadata, without inspecting generated assets."""
from pathlib import Path
import argparse,json,math,hashlib
import numpy as np
import cv2
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation,Slerp
ROOT=Path(__file__).resolve().parents[1]

def object_pose(row):
 b=np.asarray(row['projected_laser_box']);ego=np.array(row['ego_to_world']).reshape(4,4)
 local=np.eye(4);local[:3,:3]=Rotation.from_euler('z',b[6]).as_matrix();local[:3,3]=b[:3]
 return ego@local

def main():
 p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,default=ROOT/'inputs/random10_cases.json');args=p.parse_args()
 inventories=[json.loads((ROOT/f'inputs/{s}/associated_inventory.json').read_text()) for s in ['service_van','silver_sedan_03','white_sedan']]
 cases=json.loads(args.cases.read_text());summary=[]
 for case in cases:
  case=json.loads((ROOT/'inputs'/case['case_id']/'segmented_manifest.json').read_text())
  src=next(r for r in case['images'] if r['id']==case['flux_reference_id']);assert src['split']=='train'
  heldout_frames={r['frame_index'] for r in case['images'] if r['split']=='heldout'}
  inventory=next(d for d in inventories if d['context']==src['context']);lid=src['projected_laser_object_id']
  rows={r['timestamp_micros']:r for r in inventory['observations'] if r.get('projected_laser_object_id')==lid and r['frame_index'] not in heldout_frames}
  ts=src['image_timestamp_s'];before=max(t for t in rows if t*1e-6<=ts);after=min(t for t in rows if t*1e-6>=ts)
  if not 0<(after-before)*1e-6<=.25:raise ValueError('Cannot bracket source image with adjacent non-heldout labels')
  a,b=rows[before],rows[after];pa,pb=object_pose(a),object_pose(b);w=(ts-before*1e-6)/((after-before)*1e-6)
  pose=np.eye(4);pose[:3,:3]=Slerp([0,1],Rotation.from_matrix(np.stack([pa[:3,:3],pb[:3,:3]])))([w]).as_matrix()[0];pose[:3,3]=(1-w)*pa[:3,3]+w*pb[:3,3]
  native_h=(1-w)*a['projected_laser_box'][5]+w*b['projected_laser_box'][5]
  pose[:3,3]-=pose[:3,2]*native_h/2
  cam=np.array(src['camera_pose_to_world']).reshape(4,4)@np.array(src['camera_to_ego']).reshape(4,4)
  to_cv=np.eye(4);to_cv[:3,:3]=[[0,-1,0],[0,0,-1],[1,0,0]]
  view=to_cv@np.linalg.inv(cam)@pose
  intr=src['intrinsic'];crop=src['crop_box'];k=np.array([[intr[0],0,intr[2]-crop[0]],[0,intr[1],intr[3]-crop[1]],[0,0,1.]])
  rgb=np.array(Image.open(src['path']).convert('RGB'));mask=np.array(Image.open(src['mask_path']).convert('L'));h,wimg=mask.shape
  mx,my=cv2.initUndistortRectifyMap(k,np.array(intr[4:]),None,k,(wimg,h),cv2.CV_32FC1)
  und=cv2.remap(rgb,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255));alpha=cv2.remap(mask,mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
  out=ROOT/'inputs'/case['case_id']/'evaluation';out.mkdir(exist_ok=True)
  Image.fromarray(und).save(out/'source_rgb.png');Image.fromarray(alpha).save(out/'source_mask.png')
  eye=np.linalg.inv(view)[:3,3];az=math.atan2(eye[1],eye[0])
  config=dict(case_id=case['case_id'],source_id=src['id'],source_sha256=src['sha256'],source_mask_sha256=hashlib.sha256(Path(src['mask_path']).read_bytes()).hexdigest(),view_cv_from_object=view.tolist(),K_crop_undistorted=k.tolist(),image_size=[wimg,h],source_rgb=str(out/'source_rgb.png'),source_mask=str(out/'source_mask.png'),orbit_phase_rad=az,orbit_relative_azimuth_degrees=[22.5*i for i in range(16)],orbit_elevation_degrees=12,orbit_distance_m=11,orbit_focal_pixels=700,orbit_size=512,length_m=case['metric_dimensions_lwh'][0],height_m=case['metric_dimensions_lwh'][2],pose_protocol='Object world pose interpolated between adjacent native 3D labels at camera pose_timestamp; all heldout frame labels excluded. CameraImage.pose is ego-to-world; camera extrinsic applied once. Object origin moved from box center to ground. No generated-output pose fitting.',alignment_label_frames=[a['frame_index'],b['frame_index']],image_time_s=ts,interpolation_fraction=w,rolling_shutter='Single pose at image pose_timestamp; per-row rolling shutter not modeled. Metadata-assisted matched-source approximation, recorded separately from paper-exact reproduction.',best_view_selection=False)
  (out/'camera.json').write_text(json.dumps(config,indent=2))
  # Geometry-only diagnostic, independent of all generated outputs.
  l,ww,hh=case['metric_dimensions_lwh'];corners=np.array([[x,y,z,1] for z in [0,hh] for y in [-ww/2,ww/2] for x in [-l/2,l/2]])
  xyz=corners@view.T;uv=xyz[:,:3]@k.T;uv=uv[:,:2]/uv[:,2:3]
  overlay=Image.fromarray(und);draw=ImageDraw.Draw(overlay)
  for i in range(8):
   for bit in [1,2,4]:
    j=i^bit
    if j>i:draw.line([tuple(uv[i]),tuple(uv[j])],fill='lime',width=2)
  overlay.save(out/'source_cuboid_projection.jpg');summary.append(config)
 (ROOT/'docs'/('metric_source_cameras_'+args.cases.stem+'.json')).write_text(json.dumps(summary,indent=2))
 print('Prepared',len(summary),'fixed source cameras')
if __name__=='__main__':main()
