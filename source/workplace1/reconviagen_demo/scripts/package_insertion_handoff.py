"""Pack observed insertion outputs and a schema-v2 adapter; never replan or reselect assets."""
from pathlib import Path
import sys,os,json,shutil,hashlib,csv,ast,zipfile,subprocess
import numpy as np
from plyfile import PlyData
R=Path(__file__).resolve().parents[1];S=R/'scene_validation';OUT=R/'deliveries';D=OUT/'insertion_handoff_20261023';D.mkdir(parents=True,exist_ok=True)
OLD=Path('/mnt/wmj/workspace4/pipeline/traj_pipeline');sys.path.insert(0,str(OLD))
from corner_case import load_scene
from corner_case.export import load_package,_sanitize
scenes=json.loads((S/'inputs/scenes.json').read_text());cases=['random10_04_white_suv','random10_02_white_pickup'];variants=['B0','R','H','B0_real4']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(_sanitize(x),ensure_ascii=False,indent=2,allow_nan=False))
def copy(p,q):
 q.parent.mkdir(parents=True,exist_ok=True)
 if q.exists():
  if sha(p)==sha(q):return
  q.unlink()
 # Copy, do not hardlink: downstream edits must never modify original evidence.
 shutil.copy2(p,q)
def rel(p):return str(p.relative_to(D))
# Exact old function used by the existing composite renderer, without importing its GPU runtime.
old_renderer=Path('/mnt/wmj/workspace4/Recon/splatad_insert/render_package_composite.py')
fn=next(n for n in ast.parse(old_renderer.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='load_rigid')
ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(old_renderer),'exec'),ns);old_load_rigid=ns['load_rigid']
assets=[];inserts=[];audit=[];dependencies=[]
for cid in cases:
 inp=R/'inputs'/cid;manifest=json.loads((inp/'manifest.json').read_text())
 for p in inp.glob('*.json'):copy(p,D/'source_evidence'/cid/p.name)
 for p in inp.glob('*.png'):copy(p,D/'source_evidence'/cid/p.name)
 for v in variants:
  base=R/'results'/cid/f'{v}_seed42'/'asset';dst=D/'assets'/cid/v
  for name in ['canonical.ply','metadata.json','load_validation.json','render_config.json','turntable.jpg','turntable.mp4']:
   if (base/name).exists():copy(base/name,dst/name)
  ply=dst/'canonical.ply';vertices=PlyData.read(ply)['vertex'];xyz=np.column_stack([vertices[k] for k in ['x','y','z']]).astype(float)
  target=manifest['metric_dimensions_lwh'][0];scale=target/(np.percentile(xyz[:,0],99.5)-np.percentile(xyz[:,0],.5));zbase=np.percentile(xyz[:,2]*scale,.5)
  bounds=np.percentile(xyz,[.5,99.5],axis=0)*scale;size=bounds[1]-bounds[0]
  assets.append(dict(case_id=cid,variant=v,path=rel(ply),sha256=sha(ply),gaussian_count=len(xyz),metric_scale=scale,target_length_m=target,reference_width_m=manifest['metric_dimensions_lwh'][1],actual_robust_size_lwh_m=size.tolist(),scaled_base_shift_m=float(zbase),coordinate='canonical z-up, +x forward, metric scale applied once at load'))
for s in scenes:
 sid=s['scene'];scene=load_scene(S/'inputs'/f'{sid}.pkl');src_cam=S/'docs/camera_export'/sid
 cameras=json.loads((src_cam/'cameras.json').read_text());assert len(cameras['records'])==360
 w2m=np.eye(4);w2m[:3]=cameras['dataparser_world_to_model']
 cfg=Path(s['config']);ckpt=Path(cameras['checkpoint']);scenedir=D/'scenes'/sid
 copy(cfg,scenedir/'config.original.yml');copy(src_cam/'cameras.json',scenedir/'cameras.json')
 for p in [S/'inputs'/f'{sid}.pkl',S/'inputs'/f'{sid}_plan.json',S/'inputs'/f'{sid}_traj.npz']:copy(p,scenedir/p.name)
 copy(src_cam/'export.log',scenedir/'camera_export.log')
 dependencies.append(dict(scene=sid,checkpoint_original=str(ckpt),checkpoint_bundle=f'insertion_backgrounds_20261023/{sid}/nerfstudio_models/{ckpt.name}',checkpoint_sha256=sha(ckpt),checkpoint_bytes=ckpt.stat().st_size,config_original=str(cfg),config_in_core=rel(scenedir/'config.original.yml'),dataset_root='/mnt/ojc/splatad_repro/data/nuscenes_official_streams_0103_0104_0277_0795',planning_dataset_root='/mnt/public_data/nuScenes',dataset_requirement='Original nuScenes v1.0-trainval metadata/map and these scene LiDAR sweeps; existing dataparser may also load camera files. Not redistributed in this package.',source_code='/mnt/ojc/splatad_repro/src/neurad-studio',python_environment=str(R/'env_splatad/bin/python')))
 for cid in cases:
  cb=S/'results'/sid/cid
  for name in ['comparison_scene.mp4','comparison_crop.json','comparison_frame_000.jpg','comparison_frame_030.jpg','comparison_frame_059.jpg']:copy(cb/name,D/'previews'/sid/cid/name)
  if cid==cases[0]:
   bg=cb/'background'
   for p in bg.glob('*'):
    if p.is_file():copy(p,D/'background_previews'/sid/p.name)
  for v in variants:
   src=cb/v;dest=D/'runs'/sid/cid/v
   for p in src.iterdir():
    if p.is_file():copy(p,dest/p.name)
   for p in (src/'diagnostics').glob('*'):copy(p,dest/'diagnostics'/p.name)
   meta=json.loads((src/'meta.json').read_text());a=next(a for a in assets if a['case_id']==cid and a['variant']==v)
   assert meta['asset_sha256']==a['sha256'];assert abs(meta['metric_scale']-a['metric_scale'])<1e-10
   assert meta['trajectory_sha256']==sha(S/'inputs'/f'{sid}_traj.npz')
   for name in ['color_calibration.json','ground_grid.npz']:
    assert sha(src/name)==sha(src_cam/name),(sid,cid,v,name)
   if cid==cases[0]:assert np.allclose(json.loads((src/'contact_trajectory.json').read_text()),json.loads((src_cam/'contact_trajectory.json').read_text()),rtol=0,atol=1e-12)
   data=np.load(S/'inputs'/f'{sid}_traj.npz');ts=data['ts_grid'];tr=data['traj'];contact=np.array(json.loads((src/'contact_trajectory.json').read_text()));camera_epochs=np.array([x['epoch_s'] for x in cameras['records']])
   mats=[];rgs=[];positions=[];poses=[];motion=[]
   for epoch in camera_epochs:
    assert ts[0]<=epoch<=ts[-1]
    pose=np.array([np.interp(epoch,ts,tr[:,0]),np.interp(epoch,ts,tr[:,1]),np.interp(epoch,ts,np.unwrap(tr[:,2])),np.interp(epoch,ts,tr[:,3])]);yaw=pose[2]
    z=np.interp(epoch,contact[:,0],contact[:,1]);n=np.array([np.interp(epoch,contact[:,0],contact[:,k]) for k in [2,3,4]]);n/=np.linalg.norm(n);fwd=np.array([np.cos(yaw),np.sin(yaw),0.]);fwd-=fwd.dot(n)*n;fwd/=np.linalg.norm(fwd);rg=np.column_stack([fwd,np.cross(n,fwd),n]);pos=np.array([pose[0],pose[1],z]);mat=np.eye(4);mat[:3,:3]=a['metric_scale']*rg;mat[:3,3]=pos-a['scaled_base_shift_m']*rg[:,2]
    lo=max(ts[0],epoch-.02);hi=min(ts[-1],epoch+.02);duration=hi-lo
    lin=[(np.interp(hi,ts,tr[:,k])-np.interp(lo,ts,tr[:,k]))/duration for k in [0,1]];dy=np.interp(hi,ts,np.unwrap(tr[:,2]))-np.interp(lo,ts,np.unwrap(tr[:,2]));rate=np.arctan2(np.sin(dy),np.cos(dy))/duration
    mats.append(mat);rgs.append(rg);positions.append(pos);poses.append(pose);motion.append([*lin,0,rate])
   # Validate stored renderer diagnostics against independently exported camera times and reconstructed poses.
   diagnostics=json.loads((src/'visibility_diagnostics.json').read_text());rows=diagnostics if isinstance(diagnostics,list) else diagnostics.get('rows',diagnostics.get('frames'))
   if rows is None:raise ValueError(f'Unexpected diagnostics schema: {diagnostics.keys()}')
   lookup={(r['frame'],r['sensor']):r for r in rows}
   for i,c in enumerate(cameras['records']):
    d=lookup[(c['frame'],c['sensor'])];assert abs(d['epoch']-camera_epochs[i])<1e-6;assert np.allclose(d['pose'],poses[i],rtol=0,atol=1e-9);assert abs(d['ground_z']-positions[i][2])<1e-9
   np.savez_compressed(dest/'asset_poses_per_camera.npz',epoch_s=camera_epochs,frame=np.array([c['frame'] for c in cameras['records']]),sensor=np.array([c['sensor'] for c in cameras['records']]),canonical_to_nuscenes_global=np.array(mats),canonical_to_splatad_model=w2m[None]@np.array(mats),rotation_asset_to_global=np.array(rgs),ground_origin_global=np.array(positions),trajectory_xy_yaw_v=np.array(poses),linear_global_xyz_and_yaw_rate=np.array(motion))
   pkgname=f'{sid}__{s["maneuver"]}__{cid}__{v}__seed0';pkg=D/'packages'/pkgname;pkg.mkdir(parents=True,exist_ok=True)
   binding=dict(kind='rigid',uid=cid+'__'+v,asset_path=a['path'],scale=a['metric_scale'],size_lwh_m=a['actual_robust_size_lwh_m'])
   scenario=dict(schema_version=2,package=pkgname,segment=sid,maneuver=s['maneuver'],seed=s['seed'],dt=scene.dt,window=dict(start=0,length=len(ts)),params=s['params'],params_overrides=s['params'],objects=[dict(index=0,name=cid,role='attacker',category='car',action=None,is_static=False,binding=binding,metrics=s['achieved'])],generated_at='2026-10-23',handoff=dict(adapter='observed validation results -> old schema v2; not a fresh planner run',binding_path_base='handoff root: cd to root before calling unmodified old renderer',planning_collision_envelope_lw=s['planned_envelope_lw'],planning_note='Original comparison uses one conservative envelope covering all8 assets, not one independent plan per bound route',rendered_run=rel(dest),original_config=str(cfg),exported_cameras=rel(scenedir/'cameras.json'),renderer='scripts/insert_scene_validation.py',compatibility='old schema/load_rigid verified; old composite renderer has different grounding/color policy and has NOT been rerendered for this adapter'))
   write(pkg/'scenario.json',scenario);np.savez_compressed(pkg/'trajectories.npz',ego=scene.ego_states[:len(ts)],ts_grid=ts,obj0_traj=tr)
   oldscenario,arrays=load_package(pkg);assert np.array_equal(arrays['obj0_traj'],tr);b=dict(oldscenario['objects'][0]['binding']);b['asset_path']=str(D/b['asset_path']);loaded=old_load_rigid(b);assert len(loaded['means'])==a['gaussian_count'];assert np.isfinite(loaded['means']).all()
   assert abs(np.percentile(loaded['means'][:,2],.5))<1e-10
   run=dict(scene=sid,case_id=cid,variant=v,package=rel(pkg),run=rel(dest),asset=a['path'],video=rel(dest/'insert_splatad_6cam.mp4'),pose_npz=rel(dest/'asset_poses_per_camera.npz'),camera_json=rel(scenedir/'cameras.json'),frames=60,cameras=6,metric_scale=a['metric_scale'])
   inserts.append(run);audit.append(dict(scene=sid,case_id=cid,variant=v,old_loader='PASS',asset_hash='PASS',traj_hash='PASS',exported_camera_epochs_vs_original_diagnostics='PASS',reconstructed_poses_vs_original_diagnostics='PASS',color_and_ground_grid_vs_export='PASS',camera_records=len(camera_epochs)))
   print('PACKED',sid,cid,v,flush=True)
for name in ['REPORT.md','REPORT.html','REPORT.docx','REPORT.txt']:copy(S/name,D/'reports'/name)
for p in (S/'docs').glob('*'):
 if p.is_file():copy(p,D/'reports/metrics'/p.name)
for p in (S/'inputs').glob('*.json'):copy(p,D/'provenance'/p.name)
# Exact project adapter sources and old contract for inspection.
for name in ['insert_scene_validation.py','insert_scene_grounded.py','insert_scene.py','plan_scene_validation.py','run_scene_validation_task.py','export_insertion_handoff_cameras.py','package_insertion_handoff.py','audit_scene_validation.py','evaluate_scene_validation.py']:
 copy(R/'scripts'/name,D/'code'/name)
for p in (OLD/'corner_case').rglob('*.py'):copy(p,D/'code/old_traj_pipeline/corner_case'/p.relative_to(OLD/'corner_case'))
for p in [OLD/'README.md',OLD/'pyproject.toml',old_renderer,old_renderer.parent/'insert_asset_splatad.py']:copy(p,D/'code/old_contract'/p.name)
renderer_src=Path('/mnt/ojc/splatad_repro/src/neurad-studio')
for p in renderer_src.rglob('*'):
 if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and p.suffix in {'.py','.toml','.yaml','.yml','.json','.md','.cu','.cpp','.h','.cuh'}:copy(p,D/'code/training_renderer'/p.relative_to(renderer_src))
write(D/'ASSETS.json',assets);write(D/'INSERTIONS.json',inserts);write(D/'EXTERNAL_DEPENDENCIES.json',dependencies);write(D/'VALIDATION.json',dict(status='PASS',insertions=len(inserts),unique_assets=len(assets),camera_records=3*360,per_insertion_pose_records=24*360,rows=audit,limitations=['Native insertion videos are prior completed renders; package adapter was validated with the old loading functions, not rerendered through the different old composite renderer.','NuScenes raw dataset and Python environments are external dependencies.','New nonvehicle10 assets have NOT been inserted and are deliberately not mislabeled as scene results.']))
with (D/'INSERTIONS.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(inserts[0]));w.writeheader();w.writerows(inserts)
print('STAGED',D,flush=True)
