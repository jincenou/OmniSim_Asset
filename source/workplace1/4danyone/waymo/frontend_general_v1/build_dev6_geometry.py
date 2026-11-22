"""Frozen Dev6 camera/pose2x2, reliable constant reference and reversible root maps."""
from pathlib import Path
from dataclasses import replace
import argparse,json,os,sys,hashlib
import numpy as np,torch,av
from PIL import Image,ImageDraw
from safetensors.torch import load_file,save_file
G=Path(__file__).resolve().parent;W=G.parent;D=W/'frontend_quality_20261020';REPO=Path(os.environ['FDANYONE_RUNTIME']);S=W/'scene20/r20261019/scripts'
sys.path[:0]=[str(REPO),str(S)]
from fdanyone.skeleton import pipeline as skeleton
from fdanyone.skeleton.renderer import render_goliath40
from fdanyone.motion.result import MotionResult
from fdanyone.assets import resolve_regressor
from fdanyone.geometry.cameras import camera_grid,project_points
from geometry_precision import build
from root_local_geometry import separate_horizontal_root,source_camera_maps
from camera_geometry import transform_smpl_root
from reliable_reference import reliable_reference

def main():
 a=argparse.ArgumentParser();a.add_argument('sample');args=a.parse_args();torch.set_num_threads(2)
 protocol=G/'dev6_geometry_protocol.json';assert protocol.with_suffix('.sha256').read_text().split()[0]==hashlib.sha256(protocol.read_bytes()).hexdigest();cfg=json.loads(protocol.read_text());assert args.sample in cfg['subjects']
 expected=os.environ['FD_EXPECTED_GPU_UUID'].removeprefix('GPU-');assert torch.cuda.device_count()==1 and str(torch.cuda.get_device_properties(0).uuid)==expected
 cases=json.loads((G/'cohort_v1.json').read_text())['dev6'];source=Path(next(x['manifest'] for x in cases if x['sample_id']==args.sample)).parent
 motion=MotionResult.load(source/'motion');n=motion.num_frames
 refined=load_file(str(G/'motion_candidates_v2'/args.sample/'local_temporal/motion_candidates.safetensors'))
 out=G/'dev6_geometry'/args.sample;out.mkdir(parents=True,exist_ok=False)
 incam={k:v.clone() for k,v in motion.smpl_params_incam.items()};native={k:v.clone() for k,v in motion.smpl_params_global.items()};incam['betas']=refined['betas'];native['betas']=refined['betas']
 measured=np.load(D/'motion_unmasked'/args.sample/'measured_camera_maps.npz');C=measured['camera_to_waymo_world'].copy();origin=C[0,:3,3].copy();C[:,:3,3]-=origin
 Y=np.eye(4);Y[:3,:3]=[[1,0,0],[0,0,1],[0,-1,0]];C_y=Y[None]@C
 with skeleton._gvhmr_geometry_context(REPO/'third_party/GVHMR'):
  from hmr4d.utils.smplx_utils import make_smplx
  model=make_smplx('supermotion_coco17').eval()
  with torch.no_grad():pelvis=model.get_skeleton(incam['betas'])[:,0].numpy()
 def measured_params(params):
  r,t=transform_smpl_root(params['global_orient'].numpy(),params['transl'].numpy(),pelvis,C_y)
  return dict(params,global_orient=torch.tensor(r,dtype=torch.float32),transl=torch.tensor(t,dtype=torch.float32))
 repaired=dict(incam,body_pose=refined['body_pose'],global_orient=refined['incam.global_orient'],transl=refined['incam.transl'])
 cells={'native_world_original_pose':(native,incam),'measured_world_original_pose':(measured_params(incam),incam),
  'native_world_repaired_pose':(dict(native,body_pose=refined['body_pose']),dict(incam,body_pose=refined['body_pose'])),
  'measured_world_repaired_pose':(measured_params(repaired),repaired)}
 fn=build(skeleton._body_geometry);bodies={};reports={}
 for name,(world,camera) in cells.items():
  variant=replace(motion,smpl_params_global={k:v.clone() for k,v in world.items()},smpl_params_incam={k:v.clone() for k,v in camera.items()})
  geometry=fn(variant,resolve_regressor(None,model_dir=REPO/'models'),REPO/'third_party/GVHMR','cuda:0')
  geometry,ref=reliable_reference(geometry,refined['evidence_supported_visible_candidate'].numpy())
  local,maps,checks=separate_horizontal_root(geometry);local_to_incam,error=source_camera_maps(geometry,local,maps['local_to_canonical']);maps['local_to_smpl_incam']=local_to_incam
  if name.startswith('measured'):
   # Measured world->Y-up(global)->canonical->local, plus exact inverse sidecars.
   maps['relative_waymo_world_to_y_up']=Y;maps['camera_to_relative_waymo_world']=C;maps['world_origin']=origin
   full_local_to_cam=np.linalg.inv(C_y)@np.linalg.inv(maps['motion_to_local'])
   recomposed=np.einsum('tij,tnj->tni',full_local_to_cam[:,:3,:3],local.keypoints_world)+full_local_to_cam[:,None,:3,3]
   measured_error=float(np.max(np.abs(recomposed-local.keypoints_incam)));assert measured_error<2e-4,measured_error
   maps['local_to_measured_source_camera']=full_local_to_cam
  else:measured_error=None
  output=out/name;output.mkdir();variant.save(output/'motion');np.savez_compressed(output/'root_maps.npz',**maps,times=refined['times_seconds'].numpy())
  np.savez_compressed(output/'keypoints.npz',world=local.keypoints_world,incam=local.keypoints_incam,joints=local.joints_world)
  reports[name]={'reference':ref,'root_checks':checks,'source_rigid_error':error,'known_camera_recomposition_error':measured_error,'smpl_body_pose_sha256':hashlib.sha256(world['body_pose'].numpy().tobytes()).hexdigest(),'quality_approved':False,'pose_scope':'unchanged local_temporal300updates' if 'repaired' in name else 'original pose fixed mean shape'};bodies[name]=local
 # Common diagnostic cameras across all four cells; preserve frozen Goliath colors/zbuffer.
 allpoints=np.concatenate([v.keypoints_world for v in bodies.values()]);miny,maxy=allpoints[...,1].min(),allpoints[...,1].max();center=np.array([np.median(allpoints[...,0]),(miny+maxy)/2,np.median(allpoints[...,2])]);H,WW=320,176;K=np.array([[416.,0,88],[0,416.,160],[0,0,1.]])
 radius=3.
 for attempt in range(40):
  cams=camera_grid(center=center,front_direction=np.array([0,0,-1.]),K=K,image_height=H,image_width=WW,views_per_layer=24,layer_pitches=(15,),start_yaw=0,yaw_span=360,radius=radius,target_height=center[1])
  good=True
  for cam in cams:
   uv,z,_=project_points(allpoints.reshape(-1,3),cam)
   good=good and bool(((z>0)&(uv[:,0]>.04*WW)&(uv[:,0]<.96*WW)&(uv[:,1]>.04*H)&(uv[:,1]<.96*H)).all())
  if good:break
  radius*=1.1
 else:raise ValueError('Cannot fit all four conditions')
 preview_views=[0,6,12];panels=[]
 for t in range(n):
  panel=Image.new('RGB',(WW*3,(H+24)*4),'#222222');draw=ImageDraw.Draw(panel)
  for row,(name,g) in enumerate(bodies.items()):
   for column,view in enumerate(preview_views):
    uv,z,_=project_points(g.keypoints_world[t],cams[view]);image=render_goliath40(uv,z,np.ones(len(z)),canvas_height=H,canvas_width=WW,output_height=H,output_width=WW)
    panel.paste(Image.fromarray(image),(column*WW,row*(H+24)+24));draw.text((column*WW+2,row*(H+24)+5),f'{name[:8]} {"pose1" if "repaired" in name else "pose0"} v{view}',fill='white')
  panels.append(panel)
 with av.open(str(out/'four_way_skeleton.mp4'),'w') as output:
  stream=output.add_stream('libx264',rate=25);stream.width=WW*3;stream.height=(H+24)*4;stream.pix_fmt='yuv420p';stream.options={'crf':'17','threads':'2'}
  for panel in panels:
   for packet in stream.encode(av.VideoFrame.from_image(panel)):output.mux(packet)
  for packet in stream.encode():output.mux(packet)
 with av.open(str(out/'four_way_skeleton.mp4')) as video:assert sum(1 for _ in video.decode(video=0))==n
 sheet=Image.new('RGB',(WW*3*3,(H+24)*4))
 for col,t in enumerate([0,n//2,n-1]):sheet.paste(panels[t],(col*WW*3,0))
 sheet.save(out/'four_way_skeleton.jpg',quality=94)
 report={'sample':args.sample,'frames':n,'variant_reports':reports,'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),'gpu_uuid':expected,'diagnostic_camera_radius':radius,'preview_views':preview_views,'cameras':[cam.to_dict() for cam in cams],'full_24view_generation':False,'scope':'2x2 skeleton diagnostic and reusable source-consistent motions; measured root depths still estimates, no quality approval','normalization_changes':'one reliable constant yaw then horizontal root separation, no per-frame derotation'}
 (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'sample':args.sample,'state':'four_way_geometry_complete_needs_review'}))
if __name__=='__main__':main()
