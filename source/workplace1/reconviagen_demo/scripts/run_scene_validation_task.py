from pathlib import Path
import os,json,subprocess,argparse,time
R=Path(__file__).resolve().parents[1];O=R/'scene_validation'
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--case',default='random10_04_white_suv');p.add_argument('--variant',default='H');p.add_argument('--gpu',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--background',action='store_true');p.add_argument('--ground-mode',default='lower_road');a=p.parse_args()
s=next(s for s in json.loads((O/'inputs/scenes.json').read_text()) if s['scene']==a.scene);case=json.loads((R/'inputs'/a.case/'manifest.json').read_text());dest=O/'results'/a.scene/a.case/(a.variant+('_smoke_'+a.ground_mode if a.smoke else '')+('_background' if a.background else ''))
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu,PATH=str(R/'env_splatad/bin')+':/mnt/ojc/miniconda3/envs/splatad/bin:'+env['PATH'],MPLCONFIGDIR=str(R/'cache/matplotlib'),TORCH_EXTENSIONS_DIR=str(R/'cache/torch_extensions_splatad'),MAX_JOBS='4',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TORCH_HOME=str(R/'cache/torch_splatad'),HF_HOME=str(R/'cache/hf'),XDG_CACHE_HOME=str(R/'cache/xdg'),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
cmd=[str(R/'env_splatad/bin/python'),str(R/'scripts/insert_scene_validation.py'),'--config',s['config'],'--traj_npz',s['traj_npz'],'--asset_ply',str(R/'results'/a.case/f'{a.variant}_seed42/asset/canonical.ply'),'--out_dir',str(dest),'--target_len',str(case['metric_dimensions_lwh'][0]),'--reference_width',str(case['metric_dimensions_lwh'][1]),'--start_frame','60' if a.smoke else '30','--max_frames','3' if a.smoke else '60','--fps','12','--downscale','2','--ground_mode',a.ground_mode]
if a.background:cmd+=['--background_only']
else:cmd+=['--save_diagnostics']
log=O/'logs'/f'{a.scene}_{a.case}_{dest.name}.log'
with log.open('w') as f:result=subprocess.run(cmd,env=env,cwd=R,stdout=f,stderr=subprocess.STDOUT)
log.with_suffix('.exit').write_text(str(result.returncode));print(dest,result.returncode,flush=True);raise SystemExit(result.returncode)
