#!/usr/bin/env python3
"""Exercise installed cycles through actual Hasim sampling and trajectory binding."""
import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--hasim-root',type=Path,required=True);p.add_argument('--limit',type=int,default=0);p.add_argument('--report',type=Path,default=Path('cycle_validation.json'));a=p.parse_args()
sys.path.insert(0,str(a.hasim_root))
if (a.hasim_root/'src/shared/assets').is_dir():
 import importlib.util
 spec=importlib.util.spec_from_file_location('hasim',a.hasim_root/'src/__init__.py',submodule_search_locations=[str(a.hasim_root/'src')]);module=importlib.util.module_from_spec(spec);sys.modules['hasim']=module;spec.loader.exec_module(module)
import numpy as np
from hasim.shared.assets.library_v2 import AssetLibrary,load_asset
from hasim.shared.assets.human_cycle import CycleProfile,CyclicRuntimeAsset
from hasim.shared.insertion import InsertionScenario
from hasim.shared.trajectory_file import Track,TrajectoryFile
lib=AssetLibrary(a.root);rows=[];assets=[x for x in lib.assets if x.kind=='human' and x.meta.get('cycle_profile',{}).get('qualified')]
if a.limit:assets=assets[:a.limit]
for asset in assets:
 try:
  profile=CycleProfile.from_dict(asset.meta['cycle_profile']);runtime=load_asset(asset);cycle=CyclicRuntimeAsset(runtime,profile)
  for t in [0.,profile.period_s-.001,profile.period_s,profile.period_s+.001,20.5,60.]:
   f=cycle.sample(t);assert np.isfinite(f.means).all() and np.isfinite(f.scales).all();assert len(f.means)==runtime.arrays['frame_xyz'].shape[1]
  speed=max(.050001,profile.native_speed_mps) if profile.mode=='locomotion' else 0.
  track=Track('person','human',[0.,30.],[[0.,0.,0.],[30*speed,0.,0.]],[0.,0.])
  scenario=InsertionScenario(TrajectoryFile((track,)),lib,bindings={'person':asset.id},playback={'default':'cycle'})
  for t in [0.,15.,29.]:assert len(scenario.sample(t).instances)==1
  rows.append({'id':asset.id,'passed':True})
 except Exception as e:rows.append({'id':asset.id,'passed':False,'error':str(e)})
 print(asset.id,rows[-1]['passed'],flush=True)
a.report.write_text(json.dumps({'checked':len(rows),'failed':sum(not x['passed'] for x in rows),'assets':rows},ensure_ascii=False,indent=2))
if not rows or any(not x['passed'] for x in rows):raise SystemExit(1)
