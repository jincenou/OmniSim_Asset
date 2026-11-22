import json,sys,hashlib
from pathlib import Path
import numpy as np,cv2
from plyfile import PlyData
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'DriveLab'))
from asset_extraction.sign_surface import photo_plate,box_surface,splats,write
from asset_extraction.gaussians import _export_ply
from asset_extraction.export import inspect_ply
for name,base,packet in [('P07_photo_hybrid','P07_real_B0','P07_diversion'),('P08_photo_hybrid','P08_B0','P08_works_400')]:
 if len(sys.argv)>1 and name not in sys.argv[1:]: continue
 r=W/'runs'/name;d=r/'curation';d.mkdir(parents=True,exist_ok=True);(r/'generation').mkdir(exist_ok=True);g=json.loads((W/'runs'/base/'generation/generation.json').read_text());(r/'generation/generation.json').write_text(json.dumps(g,indent=2));im=cv2.cvtColor(cv2.imread(str(W/'inputs'/packet/'source.png')),cv2.COLOR_BGR2RGB)
 if name.startswith('P07'):
  rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
  _export_ply(W/'runs'/base/'curation/axis.ply',d/'oriented.ply',rotation,np.zeros(3));v=PlyData.read(str(d/'oriented.ply'))['vertex'].data
  # Keep the original-photo B0 post, replace damaged base with explicit solid completion.
  support=v[(v['z']>=.07)&(v['z']<.42)&(np.abs(v['y'])<.05)&(v['opacity']>0)].copy()
  lower=box_surface([-.19,-.19,0],[.19,.19,.08],[.095,.10,.105],step=.0025)
  factor=1-.35*lower['z']/.08;lower['x']*=factor;lower['y']*=factor
  for i in range(3):lower[f'scale_{i}']=np.log(.0017)
  quad=[[63,66],[188,75],[190,199],[67,219]];plate,tex,H=photo_plate(im,quad,.70,.53,.41,corner_radius=.035,x=.025,rear_rgb=(.12,.12,.12))
  parts=[support,lower,plate];height_m=1.4;quads=[quad];textures=[tex];transforms=[H.tolist()]
  method='Real-only B0 post retained. Damaged base replaced by explicitly assumed solid tapered base; original PDF-photo legend mapped to separate thin plate.'
  defects=['Low-resolution original 269x403; face letters soften at close range','Exact size not established; preview total height 1.40 m is provisional','Base shape repaired with geometric prior; small original slots, connector details and underside not reconstructed','Unobserved rear neutral black; no mirrored arrow']
 else:
  v=PlyData.read(str(W/'runs'/base/'curation/axis.ply'))['vertex'].data;support=v[(v['z']<.305)&(v['opacity']>0)].copy();ground=float(support['z'].min());support['z']-=ground
  tri=[[557,63],[724,366],[393,399]];quad=[[400,422],[723,387],[726,522],[404,568]]
  p1,t1,H1=photo_plate(im,tri,.49,.44,.56,triangle=True,x=.006,rear_rgb=(.12,.12,.12))
  p2,t2,H2=photo_plate(im,quad,.49,.22,.31,corner_radius=.04,x=.006,rear_rgb=(.12,.12,.12))
  post=box_surface([-.036,-.018,.28],[-.006,.018,.94],[.90,.64,.035],step=.002)
  parts=[support,post,p1,p2];height_m=1.55;quads=[tri,quad];textures=[t1,t2];transforms=[H1.tolist(),H2.tolist()]
  method='B0 weighted base and lower post retained; upper hidden post completed; source worker triangle and 400 yds panel replace corrupted neural surfaces independently.'
  defects=['Exact source size unresolved; 1.55 m total height is provisional','Hidden upper post and connectors completed, not measured','Both unknown rear plates neutral black; source illumination retained in front texture','Triangular plate rectified with three-point affine approximation; rounded corners simplified']
 write(d/'canonical.ply',parts)
 for i,tex in enumerate(textures):cv2.imwrite(str(d/f'rectified_source_plate_{i}.png'),cv2.cvtColor(tex,cv2.COLOR_RGB2BGR))
 cv2.imwrite(str(d/'rectified_source_plate.png'),cv2.cvtColor(textures[0],cv2.COLOR_RGB2BGR))
 v=PlyData.read(str(d/'canonical.ply'))['vertex'].data;p=np.column_stack([v[k] for k in ['x','y','z']]);height=float(p[:,2].max());post=dict(method=method,parent_run=base,parent_raw_sha256=g['assets'][0]['raw_sha256'],source=str(W/'inputs'/packet/'source.png'),source_sha256=hashlib.sha256((W/'inputs'/packet/'source.png').read_bytes()).hexdigest(),source_quads=quads,transforms=transforms,defects=defects,not_a_matched_neural_result=True)
 (d/'postprocess.json').write_text(json.dumps(post,indent=2))
 c=dict(canonical_path=str(d/'canonical.ply'),sha256=hashlib.sha256((d/'canonical.ply').read_bytes()).hexdigest(),up='+Z',units='normalized',metric_scale_baked=False,scale_reference_height=height,size_xyz=np.ptp(p,axis=0).tolist(),heading=dict(front='+X',azimuth_deg=0,confidence=1.,source=str(d/'rectified_source_plate.png'),verification=dict(front='A',confidence=1.)),heading_verified=True,grounding='minimum support center at z=0; one metric scale only',support_contact=dict(min_center_z_after=float(p[:,2].min())),metric_preview=dict(height_m=height_m,width_m=float(np.ptp(p[:,1])*height_m/height),depth_m=float(np.ptp(p[:,0])*height_m/height),isotropic_factor=height_m/height),postprocess=post,validation=inspect_ply(d/'canonical.ply'))
 (d/'canonical.json').write_text(json.dumps(c,indent=2));print(name,c['metric_preview'])
