from pathlib import Path
import sys,os,json,pickle,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'scene_validation';sys.path.insert(0,'/mnt/wmj/workspace4/pipeline/traj_pipeline')
os.environ['MPLCONFIGDIR']=str(R/'cache/matplotlib');os.environ['PYTHONDONTWRITEBYTECODE']='1'
from nuscenes import NuScenes
from corner_case.scene.extract_nuscenes import extract_scene
from corner_case import PlanContext,Window,REGISTRY,load_scene
from corner_case.assets import Reservation
from corner_case.planning.objects import AssetQuery
from corner_case.planning.metrics import vehicle_min_gap,agent_min_gap
cases=['random10_04_white_suv','random10_02_white_pickup'];variants=['B0','R','H','B0_real4'];dims=[]
for c in cases:
 for v in variants:
  m=json.loads((R/'results'/c/f'{v}_seed42/asset/metadata.json').read_text());size=m['canonical']['size_xyz'];l=json.loads((R/'inputs'/c/'manifest.json').read_text())['metric_dimensions_lwh'][0];dims.append((l,l*size[1]/size[0]))
envelope=tuple((np.max(dims,axis=0)+.1).tolist());print('ENVELOPE',envelope,flush=True)
class BoundPool:
 def reserve(self,category,action=None,fallback_lw=None):return Reservation(uid='comparison_envelope',category=category,lw=envelope,height_m=2.5,query=AssetQuery(category=category,uid='comparison_envelope'),entry=None)
configs={}
for sid in ['0104','0103','0277','0795']:
 files=list(Path(f'/mnt/ojc/splatad_repro/outputs/full_scene{sid}').glob('unnamed/splatad/*/config.yml'));configs['scene-'+sid]=str(files[0])
nusc=NuScenes(version='v1.0-trainval',dataroot='/mnt/public_data/nuScenes',verbose=False);scene_list=[];attempts=[]
for name,cfg in configs.items():
 p=O/'inputs'/f'{name}.pkl'
 if p.exists():data=pickle.loads(p.read_bytes())
 else:
  data=extract_scene(nusc,next(s for s in nusc.scene if s['name']==name),hz=10);p.write_bytes(pickle.dumps(data))
 if data['map'].get('synthetic'):attempts.append(dict(scene=name,status='rejected_synthetic_map'));continue
 scene=load_scene(str(p));candidates=[('V5_merge_in',dict(allow_ego_collision=False,target_ttc=1.5,trigger_dist_m=25,dist_ahead_m=25,accel_mps2=1.5,merge_time_s=3.0,init_speed_mps=0.0)),('V1_cut_in',dict(allow_ego_collision=False,target_gap_m=5.0)),('V2_hard_brake',dict(allow_ego_collision=False,target_ttc=2.0)),('V3_slow_block',dict(allow_ego_collision=False,target_ttc=3.0,speed_mps=3.0,init_gap_m=25.0)),('V6_pull_out',dict(allow_ego_collision=False,target_ttc=2.0))]
 selected=None
 for maneuver,params in candidates:
  if maneuver not in [d.id for d in REGISTRY.by_category(__import__('corner_case.taxonomy',fromlist=['Category']).Category.VEHICLE)]:continue
  for seed in range(1):
   try:
    ctx=PlanContext(scene=scene,window=Window(0,min(80,scene.num_frames)),rng=np.random.default_rng(seed),params=REGISTRY[maneuver].params_cls.from_dict(params),assets=BoundPool());obj=REGISTRY[maneuver].planner(ctx)[0];traj=obj.traj
    eg=vehicle_min_gap(scene.ego_states[:len(traj)],traj,actor_lw=envelope);ag=agent_min_gap(traj,scene.logged_agents(),traj_lw=envelope)
    if min(eg,ag)<.25:raise ValueError(f'Envelope gap insufficient ego={eg} agent={ag}')
    selected=dict(scene=name,config=cfg,scene_pkl=str(p),maneuver=maneuver,params=params,seed=seed,planned_envelope_lw=envelope,achieved=obj.achieved.to_dict(),audited_capsule_gap_ego_m=eg,audited_capsule_gap_agents_m=ag,map_synthetic=False)
    dest=O/'inputs'/f'{name}_traj.npz';np.savez(dest,traj=traj,ts_grid=np.asarray(data['timestamps'])[:len(traj)],meta=json.dumps(selected));selected['traj_npz']=str(dest);selected['traj_sha256']=hashlib.sha256(dest.read_bytes()).hexdigest();selected['active_frame_range']=np.where(np.isfinite(traj[:,0]))[0][[0,-1]].tolist()
    attempts.append(dict(scene=name,maneuver=maneuver,seed=seed,status='selected'));break
   except Exception as e:attempts.append(dict(scene=name,maneuver=maneuver,seed=seed,status='rejected',reason=str(e)));print(name,maneuver,seed,str(e),flush=True)
  if selected:break
 if selected:
  scene_list.append(selected);(O/'inputs'/f'{name}_plan.json').write_text(json.dumps(selected,indent=2));print('PLANNED',name,selected,flush=True)
 (O/'inputs/planning_attempts.json').write_text(json.dumps(attempts,indent=2));(O/'inputs/scenes.json').write_text(json.dumps(scene_list,indent=2))
 if len(scene_list)==3:break
print('PLANNING COMPLETE',len(scene_list),flush=True)
