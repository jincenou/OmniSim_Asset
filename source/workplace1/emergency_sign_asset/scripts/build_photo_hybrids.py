import json,sys,hashlib,shutil
from pathlib import Path
import numpy as np
import cv2
from plyfile import PlyData
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'DriveLab'))
from asset_extraction.sign_surface import photo_plate,box_surface,sloping_leg,write
from asset_extraction.gaussians import _export_ply
from asset_extraction.export import inspect_ply

for name,base,packet in [('V05_photo_hybrid','V05_B0','V05_road_work'),('P01_photo_prior','P01_H0','P01_road_closed')]:
 if len(sys.argv)>1 and name not in sys.argv[1:]: continue
 r=W/'runs'/name;d=r/'curation';d.mkdir(parents=True,exist_ok=True)
 g=json.loads((W/'runs'/base/'generation/generation.json').read_text())
 (r/'generation').mkdir(exist_ok=True)
 # The parent inference receipt remains unchanged; this separate stage is explicit.
 (r/'generation/generation.json').write_text(json.dumps(g,indent=2))
 image=cv2.cvtColor(cv2.imread(str(W/'inputs'/packet/'source.png')),cv2.COLOR_BGR2RGB)
 if name.startswith('V05'):
  quad=[[485,54],[733,291],[509,524],[269,310]]
  plate,texture,H=photo_plate(image,quad,.774,.748,.245,diamond=True,x=.014)
  _export_ply(W/'runs'/base/'curation/axis.ply',d/'support_oriented.ply',np.array([[0,-1,0],[1,0,0],[0,0,1]]),np.zeros(3))
  v=PlyData.read(str(d/'support_oriented.ply'))['vertex'].data
  rgb=.5+.28209479177387814*np.column_stack([v[f'f_dc_{i}'] for i in range(3)])
  residual_orange=(rgb[:,0]>rgb[:,1]*1.6)&(rgb[:,0]>.32)&(v['z']>.18)
  support=v[(v['z']<.255)&(v['opacity']>0)&~residual_orange&((v['z']<.20)|(np.abs(v['y'])<.026))].copy()
  ground=float(support['z'].min());support['z']-=ground;plate['z']-=ground
  parts=[support,plate];metric_height=1.6
  method='Original-photo plate replaces distorted B0 plate; B0 base/upright below z=0.255 retained and grounded.'
  defects=['Exact source unit size (36/48 inch variant) unresolved; 1.6 m total height is an explicit preview assumption','Rear plate, thickness and rear connectors unobserved; neutral rear is geometric completion','Minor source grass texture remains on weighted base edge']
 else:
  quad=[[117,85],[902,78],[855,628],[166,661]]
  plate,texture,H=photo_plate(image,quad,.600,.400,.325,x=.018)
  parts=[plate,box_surface([-.012,-.305,0],[.012,-.285,.750],[.065,.07,.075]),box_surface([-.012,.285,0],[.012,.305,.750],[.065,.07,.075]),box_surface([-.012,-.305,.730],[.012,.305,.750],[.065,.07,.075]),box_surface([-.012,-.305,.305],[.012,.305,.325],[.065,.07,.075])]
  parts.extend([sloping_leg(y) for y in [-.275,.275]])
  parts.append(box_surface([-.32,-.285,.04],[-.30,.285,.06],[.065,.07,.075]))
  # Visible retaining clips, approximate geometry; exact source pixels stay on panel.
  for y,z in [(-.26,.725),(.26,.725),(-.26,.325),(.26,.325),(-.30,.49),(.30,.49)]:
   parts.append(box_surface([.02,y-.008,z-.012],[.027,y+.008,z+.012],[.67,.69,.69],step=.002))
  metric_height=.750;ground=0.
  method='Original-photo panel on explicit geometric prior for connected folding frame; H reconstruction rejected for disconnected rear legs. This is not a neural reconstruction result.'
  defects=['Frame is a parameterized completion, not measured 3D geometry; rear foot depth 0.32 m assumed','Manufacturer overall 0.610 x 0.750 m and plate 0.600 x 0.400 m applied; source photo proportions disagree and need confirmation','Unknown rear panel rendered neutral; approximate clips/hinges and cross brace']
 write(d/'canonical.ply',parts)
 cv2.imwrite(str(d/'rectified_source_plate.png'),cv2.cvtColor(texture,cv2.COLOR_RGB2BGR))
 v=PlyData.read(str(d/'canonical.ply'))['vertex'].data;p=np.column_stack([v[k] for k in ['x','y','z']]);height=float(p[:,2].max())
 evidence=dict(method=method,parent_run=base,parent_raw_sha256=g['assets'][0]['raw_sha256'],source=str(W/'inputs'/packet/'source.png'),source_sha256=hashlib.sha256((W/'inputs'/packet/'source.png').read_bytes()).hexdigest(),source_quad=quad,homography=H.tolist(),rear='neutral unobserved completion, no front text',defects=defects,not_a_matched_neural_result=True)
 (d/'postprocess.json').write_text(json.dumps(evidence,indent=2))
 c=dict(canonical_path=str(d/'canonical.ply'),sha256=hashlib.sha256((d/'canonical.ply').read_bytes()).hexdigest(),up='+Z',raw_up='mixed; see postprocess',units='normalized',metric_scale_baked=False,scale_reference_height=height,
 heading=dict(front='+X',azimuth_deg=0,confidence=1.,source=str(d/'rectified_source_plate.png'),verification=dict(front='A',confidence=1.)),heading_verified=True,
 grounding='minimum support Gaussian center at zero; Gaussian tails have finite support below contact plane',support_contact=dict(min_center_z_after=float(p[:,2].min()),old_min_z=ground),
 metric_preview=dict(height_m=metric_height,width_m=float(np.ptp(p[:,1])*metric_height/height),depth_m=float(np.ptp(p[:,0])*metric_height/height),isotropic_factor=metric_height/height),postprocess=evidence,validation=inspect_ply(d/'canonical.ply'))
 (d/'canonical.json').write_text(json.dumps(c,indent=2));print(name,c['metric_preview'])
