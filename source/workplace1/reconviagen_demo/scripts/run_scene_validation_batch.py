from pathlib import Path
import sys,json,os,argparse,gc,contextlib,traceback,time
import torch
R=Path(__file__).resolve().parents[1];O=R/'scene_validation';p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args()
import insert_scene_validation as runner
import nerfstudio.utils.eval_utils as ev
import nerfstudio.data.datasets.base_dataset as bd
original_eval=ev.eval_setup;original_image=bd.InputDataset.get_numpy_image;cache={}
def cached_eval(*args,**kwargs):
 key=str(args[0])
 if key not in cache:cache[key]=original_eval(*args,**kwargs)
 else:
  # Model parameters are never mutated by insertion; append occurs only in rasterization arguments.
  model=cache[key][1].model;model.load_state_dict(model.state_dict(),strict=True)
 return cache[key]
ev.eval_setup=cached_eval
original_lidar=runner.load_scene_lidar_global;lidars={}
def cached_lidar(*args):
 if args not in lidars:lidars[args]=original_lidar(*args)
 return lidars[args]
runner.load_scene_lidar_global=cached_lidar
scene=next(s for s in json.loads((O/'inputs/scenes.json').read_text()) if s['scene']==a.scene)
for caseid in ['random10_04_white_suv','random10_02_white_pickup']:
 c=json.loads((R/'inputs'/caseid/'manifest.json').read_text())
 for variant in ['B0','R','H','B0_real4']+(['background'] if caseid=='random10_04_white_suv' else []):
  out=O/'results'/a.scene/caseid/variant
  if (out/'meta.json').exists():continue
  sourcevariant='H' if variant=='background' else variant
  sys.argv=['insert_scene_validation.py','--config',scene['config'],'--traj_npz',scene['traj_npz'],'--asset_ply',str(R/'results'/caseid/f'{sourcevariant}_seed42/asset/canonical.ply'),'--out_dir',str(out),'--target_len',str(c['metric_dimensions_lwh'][0]),'--reference_width',str(c['metric_dimensions_lwh'][1]),'--start_frame','30','--max_frames','60','--fps','12','--downscale','2','--ground_mode','lower_road']
  sys.argv+=['--background_only'] if variant=='background' else ['--save_diagnostics']
  bd.InputDataset.get_numpy_image=original_image
  log=O/'logs'/f'{a.scene}_{caseid}_{variant}.log'
  print('START',caseid,variant,flush=True)
  try:
   with log.open('w') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):runner.main()
   log.with_suffix('.exit').write_text('0')
  except Exception:
   log.with_suffix('.failure.txt').write_text(traceback.format_exc());raise
  gc.collect();torch.cuda.empty_cache();print('DONE',caseid,variant,flush=True)
print('SCENE COMPLETE',a.scene,flush=True)
