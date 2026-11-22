from pathlib import Path
import json,csv
import numpy as np
from safetensors import safe_open
from motion_evidence import uncertain_intervals,resample_measured_camera
ROOT=Path(__file__).resolve().parent
rows=[]
for case in sorted((ROOT/'motion_unmasked').iterdir()):
 old=Path(json.loads((case/'input_verification.json').read_text())['baseline'])
 for variant,path in [('masked',old/'motion'),('unmasked',case/'motion_pair/calibrated_moving')]:
  with safe_open(path/'motion.safetensors',framework='np') as f:
   pose=f.get_tensor('smpl_params_incam.body_pose').reshape(-1,21,3);kp=f.get_tensor('observed_keypoints_2d')
  knee=np.rad2deg(np.linalg.norm(pose[:,[3,4]],axis=-1));e=uncertain_intervals(kp,np.arange(len(pose))/25)
  (case/f'{variant}_evidence.json').write_text(json.dumps(e,indent=2,allow_nan=False)+'\n')
  rows.append(dict(sample=case.name,variant=variant,left_knee_rotation_peak_deg=float(knee[:,0].max()),right_knee_rotation_peak_deg=float(knee[:,1].max()),
   low_left_ankle_conf_frames=int((kp[:,15,2]<.3).sum()),low_right_ankle_conf_frames=int((kp[:,16,2]<.3).sum()),uncertain_limb_intervals=len(e['intervals'])))
 native=np.load(old/'input/native_cameras.npz');mapping=json.loads((old/'input/frame_mapping.json').read_text())
 # Source times are exposure-relative, not arbitrary frame-number displacement.
 q=np.array([x['source_relative_exposure_seconds'] for x in mapping['frames']])
 t=native['image_center_timestamps_s'];r=resample_measured_camera(t-t[0],native['image_center_cv_camera_to_world'],q)
 np.savez_compressed(case/'measured_camera_maps.npz',camera_to_waymo_world=r,world_to_camera=np.linalg.inv(r),relative_exposure_seconds=q)
with (ROOT/'motion_ablation.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(ROOT/'motion_ablation.json').write_text(json.dumps({'rows':rows,'scope':'Local knee rotation magnitudes are model outputs, not pose error. Confidence gaps are missing-evidence flags. Full measured camera maps prepared; not yet applied in pose optimization. No method declared successful.'},indent=2)+'\n')
print('Eight-person numerical evidence and measured translation+rotation maps saved; no automatic quality pass.')
