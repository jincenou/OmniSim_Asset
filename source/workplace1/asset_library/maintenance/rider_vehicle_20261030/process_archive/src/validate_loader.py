"""Integration check through the real downstream loader; independent pose invariants."""
import importlib.util,sys,json
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parents[1];path='/mnt/ojc/workplace1/reconviagen_demo/scripts/insert_scene_validation.py';spec=importlib.util.spec_from_file_location('insertion',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
name=sys.argv[1];r=W/'runs'/name;length=2.267 if name.startswith('nuscenes') else 1.949
a=m.load_canonical_ply(str(r/'selected/canonical.ply'),target_len=length);v=a['means'];actual=np.diff(np.quantile(v[:,0],[.005,.995]))[0];assert np.isclose(actual,length,rtol=1e-6)
assert all(np.isfinite(x).all() for x in a.values());assert np.all(a['scales']>0);assert np.allclose(np.linalg.norm(a['quats'],axis=1),1,atol=1e-6)
idx=np.linspace(0,len(v)-1,128,dtype=int);pts=v[idx];baseline=np.linalg.norm(pts[1:]-pts[:-1],axis=1);errors={}
for mode in ['static','straight','gentle_turn']:
 d=np.load(W/'configs'/('scene0103_'+mode+'.npz'));tr=d['traj'];maxerr=0
 for pose in tr[[0,40,-1]]:
  x,y,yaw=pose[:3];co,si=np.cos(yaw),np.sin(yaw);rot=np.array([[co,-si,0],[si,co,0],[0,0,1]])
  posed=pts@rot.T+np.array([x,y,0]);err=np.max(abs(np.linalg.norm(posed[1:]-posed[:-1],axis=1)-baseline));maxerr=max(maxerr,float(err))
  q=m.quat_left_mul(m.rotmat_to_quat(rot),a['quats'][idx]);assert np.allclose(np.linalg.norm(q,axis=1),1,atol=1e-6)
 assert maxerr<1e-9;errors[mode]=maxerr
out=dict(passed=True,loader=path+':load_canonical_ply',point_count=len(v),target_length_m=length,loaded_length_m=float(actual),scale_applied_once=float(a['metric_scale']),pose_pairwise_distance_max_error_m=errors,scope='Numerical downstream loading and rigid transforms; actual scene visual results are separate')
(r/'selected/load_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
