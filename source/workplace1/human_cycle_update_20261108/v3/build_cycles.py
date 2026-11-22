import os,sys,json,hashlib,traceback,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
CONFIG=argparse.ArgumentParser(description='Generate and numerically qualify source-bound human cycles')
CONFIG.add_argument('--root',type=Path,required=True)
CONFIG.add_argument('--hasim-root',type=Path,required=True)
CONFIG.add_argument('--output',type=Path,required=True)
CONFIG.add_argument('--jobs',type=int,default=4)
CONFIG.add_argument('--start',type=int,default=0)
CONFIG.add_argument('--retry-failed',action='store_true')
CONFIG.add_argument('--asset-id',action='append',default=[])
ARGS=CONFIG.parse_args()
HASIM=ARGS.hasim_root.resolve()
sys.path.insert(0,str(HASIM))
if (HASIM/'src/shared/assets').is_dir():
 import importlib.util
 spec=importlib.util.spec_from_file_location('hasim',HASIM/'src/__init__.py',submodule_search_locations=[str(HASIM/'src')])
 module=importlib.util.module_from_spec(spec);sys.modules['hasim']=module;spec.loader.exec_module(module)
import numpy as np
from hasim.shared.assets.library_v2 import AssetLibrary,load_asset
from hasim.shared.assets.human_cycle import CycleProfile,CyclicRuntimeAsset,runtime_fingerprint
BASE=ARGS.output.resolve();BASE.mkdir(parents=True,exist_ok=True)
ROOT=ARGS.root.resolve()
LIMITS={'body_rms_m':.06,'foot_rms_m':.09,'body_max_m':.10,'foot_max_m':.15,'seam_velocity_error_mps':.025}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def positions(ar,times,idx):
 ts=ar['timestamps']-ar['timestamps'][0];out=[]
 for t in times:
  hi=min(int(np.searchsorted(ts,t,side='right')),len(ts)-1);lo=max(0,hi-1);u=(t-ts[lo])/(ts[hi]-ts[lo]) if hi!=lo else 0
  x=(1-u)*ar['frame_xyz'][lo,idx].astype(float)+u*ar['frame_xyz'][hi,idx].astype(float)
  x=x@ar['asset_to_standard'][:3,:3].T+ar['asset_to_standard'][:3,3]
  x[:,:2]-=(1-u)*ar['root_xy'][lo]+u*ar['root_xy'][hi]
  yaw=-((1-u)*ar['root_yaw'][lo]+u*ar['root_yaw'][hi]);c,s=np.cos(yaw),np.sin(yaw)
  x[:,:2]=x[:,:2]@np.array([[c,s],[-s,c]]);x[:,2]+=float(ar['ground_offset']);out.append(x)
 return np.array(out)
def metrics(ar,start,end,width,idx):
 a=positions(ar,start+np.linspace(-width,width,5),idx);b=positions(ar,end+np.linspace(-width,width,5),idx)
 w=ar['opacities'][idx].astype(float);w/=w.sum();z=np.median((a+b)/2,axis=0)[:,2];foot=z<np.quantile(z,.02)+.30;fw=w[foot]/w[foot].sum()
 d=np.sum((a-b)**2,axis=-1);body=np.sqrt(d@w);feet=np.sqrt(d[:,foot]@fw)
 return dict(body_rms_m=float(np.sqrt(np.mean(body**2))),foot_rms_m=float(np.sqrt(np.mean(feet**2))),body_max_m=float(max(body)),foot_max_m=float(max(feet)))
def one(meta_path):
 m=json.loads(Path(meta_path).read_text());identity=m['id'];out=BASE/'reports'/f'{identity}.json'
 if out.exists():
  previous=json.loads(out.read_text())
  if not ARGS.retry_failed or previous['status']=='qualified_numeric':return previous['status']
 row={'id':identity,'folder':Path(meta_path).parent.name,'motion_type':m['motion_type'],'original_metadata_sha256':sha(meta_path)}
 try:
  lib=AssetLibrary(ROOT);a=lib.get(identity);runtime=load_asset(a);ar=runtime.arrays
  row['runtime_file_sha256']=sha(a.path);row['source_sha256']=runtime_fingerprint(runtime)
  if m.get('motion_classification',{}).get('cycle_review_blocked'):raise ValueError('classification review blocked cycling due to reconstruction quality')
  if a.motion_type not in {'walking','jogging','running','standing','gesturing','standing_or_small_gesture'}:raise ValueError('unclassified source action; cannot assign locomotion/stationary safely')
  if not a.playback_capabilities.get('heading_normalized'):raise ValueError('missing heading normalization capability')
  mode='locomotion' if a.motion_type in {'walking','jogging','running'} else 'stationary'
  if mode=='locomotion' and (a.native_speed_mps is None or a.native_speed_mps<=.02):raise ValueError('missing positive stride speed')
  visible=np.flatnonzero(ar['opacities']>.2)
  if len(visible)<100:raise ValueError('insufficient visible geometry')
  idx=visible[np.linspace(0,len(visible)-1,min(512,len(visible)),dtype=int)]
  ts=ar['timestamps']-ar['timestamps'][0];tm=(ts[:-1]+ts[1:])/2;width=.16
  eligible=tm[(tm>.15)&(tm<ts[-1]-.15)]
  mids=positions(ar,eligible,idx);w=ar['opacities'][idx].astype(float);w/=w.sum();z=np.median(mids[:,:,2],axis=0);foot=z<np.quantile(z,.02)+.30;fw=w[foot]/w[foot].sum();candidates=[]
  for i,s in enumerate(eligible):
   js=np.flatnonzero((eligible-s>=.64)&(eligible-s<=min(2.4,ts[-1]-.3)))
   if not len(js):continue
   delta=np.sum((mids[js]-mids[i])**2,axis=-1)
   costs=np.sqrt(delta@w)/.06+np.sqrt(delta[:,foot]@fw)/.09
   candidates.extend((float(cost),float(s),float(eligible[j])) for cost,j in zip(costs,js))
  candidates.sort();scored=[]
  for _,s,e in candidates[:100]:
   for width in [.12,.16,.20,.24]:
    if s<width+.02 or e+width>runtime.duration_s-.02 or e-s<=2*width:continue
    v=metrics(ar,s,e,width,idx)
    scored.append((max(v[k]/LIMITS[k] for k in v),s,e,width))
  scored.sort()
  if not scored:raise ValueError('no valid interior cycle interval')
  if m.get('cycle_profile',{}).get('qualified'):
   old=m['cycle_profile'];options=[(0.,old['start_s'],old['end_s'],old['blend_half_window_s'])]
  else:options=scored[:10]
  full=[]
  for _,s,e,width in options:
   candidate_metrics=metrics(ar,s,e,width,visible)
   cost=max(candidate_metrics[k]/LIMITS[k] for k in candidate_metrics)
   full.append((cost,s,e,width,candidate_metrics))
   if cost<=1:break
  _,s,e,width,measured=min(full,key=lambda x:x[0])
  p=CycleProfile(identity,s,e,width,runtime.duration_s,row['source_sha256'],mode,a.native_speed_mps,qualified=True,diagnostics={})
  cycle=CyclicRuntimeAsset(runtime,p)
  eps=.001;frames=[cycle.sample(p.period_s+d) for d in [-eps,0,eps]]
  ww=ar['opacities'][visible].astype(float);ww/=ww.sum()
  vm=(frames[2].means[visible].astype(float)-2*frames[1].means[visible]+frames[0].means[visible])/eps
  measured['seam_velocity_error_mps']=float(np.sqrt(np.sum(vm**2,axis=1)@ww))
  for t in [0.,p.period_s*3+.1,20.5,60.]:
   f=cycle.sample(t);assert len(f.means)==ar['frame_xyz'].shape[1] and np.isfinite(f.means).all() and np.isfinite(f.scales).all() and np.all(f.scales>0)
  from hasim.shared.insertion import InsertionScenario
  from hasim.shared.trajectory_file import Track,TrajectoryFile
  speed=max(.050001,p.native_speed_mps) if mode=='locomotion' else 0.
  track=Track('person','human',[0.,30.],[[0.,0.,0.],[30*speed,0.,0.]],[0.,0.])
  from hasim.shared.assets.human_cycle import CycleCatalog
  InsertionScenario(TrajectoryFile((track,)),lib,bindings={'person':identity},playback={'default':'cycle'},cycle_catalog=CycleCatalog([p]))
  failed=[k for k,limit in LIMITS.items() if measured[k]>limit]
  preserved=bool(m.get('cycle_profile'))
  qualified=not failed
  diagnostics={'qualification_basis':'full_visible_geometry_and_actual_runtime_seam_test','visual_review_complete':bool(m.get('cycle_profile',{}).get('diagnostics',{}).get('visual_review_complete',False)),'contact_verified':False,'anti_skating_verified':False,'metrics':measured,'limits':LIMITS,'failed_checks':failed,'preserved_existing_profile':preserved,'search_version':'adaptive_windows_v2','runtime_samples_seconds':[0,round(p.period_s*3+.1,5),20.5,60.]}
  profile=p.to_dict();profile['qualified']=qualified;profile['diagnostics']=diagnostics
  row.update(status='qualified_numeric' if qualified else 'needs_review',profile=profile,metrics=measured,failed_checks=failed)
 except Exception as exc:row.update(status='unsupported',error=str(exc),traceback=traceback.format_exc())
 out.write_text(json.dumps(row,ensure_ascii=False,indent=2));return row['status']
if __name__=='__main__':
 args=ARGS;(BASE/'reports').mkdir(exist_ok=True)
 paths=sorted(ROOT.glob('*/asset.json'))[args.start:]
 if args.asset_id:paths=[p for p in paths if json.loads(p.read_text())['id'] in args.asset_id]
 counts={}
 with ProcessPoolExecutor(args.jobs) as pool:
  futures={pool.submit(one,str(p)):p for p in paths}
  for i,f in enumerate(as_completed(futures)):
   s=f.result();counts[s]=counts.get(s,0)+1
   if (i+1)%10==0:print(i+1,len(paths),counts,flush=True)
 print(counts,flush=True)
