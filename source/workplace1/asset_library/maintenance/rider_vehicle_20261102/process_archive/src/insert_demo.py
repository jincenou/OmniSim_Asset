"""Use the existing real SplatAD scene loader and renderer under GPU lease."""
import sys,json,os,asyncio,fcntl,hashlib
from pathlib import Path
W=Path(__file__).resolve().parents[1];R=Path('/mnt/ojc/workplace1/reconviagen_demo');sys.path.insert(0,str(W/'src/DriveLab'))
from asset_extraction.generation import GpuPool
async def run_insertion():
 name=sys.argv[1];mode=sys.argv[2];gpu_index=int(sys.argv[3]) if len(sys.argv)>3 else 1;run=W/'runs'/name;dest=run/('insertion_'+mode);s=next(s for s in json.loads((R/'scene_validation/inputs/scenes.json').read_text()) if s['scene']=='scene-0103')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu_index),PATH=str(R/'env_splatad/bin')+':/mnt/ojc/miniconda3/envs/splatad/bin:'+os.environ['PATH'],MPLCONFIGDIR=str(W/'runs/cache/matplotlib'),TORCH_EXTENSIONS_DIR=str(R/'cache/torch_extensions_splatad'),MAX_JOBS='4',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TORCH_HOME=str(R/'cache/torch_splatad'),HF_HOME=str(R/'cache/hf'),XDG_CACHE_HOME=str(R/'cache/xdg'),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
 length=json.loads((W/'configs/metric_priors.json').read_text())[name]['length_m']
 traj=W/'configs'/('scene0103_'+mode+'.npz')
 cmd=[str(R/'env_splatad/bin/python'),str(R/'scripts/insert_scene_validation.py'),'--config',s['config'],'--traj_npz',str(traj) if traj.exists() else s['traj_npz'],'--asset_ply',str(run/'selected/canonical.ply'),'--out_dir',str(dest),'--target_len',str(length),'--reference_width','.8','--start_frame','60','--max_frames','3' if mode=='smoke' else '24','--fps','12','--downscale','2','--ground_mode','lower_road','--save_diagnostics']
 pool=GpuPool([gpu_index]);await pool.acquire('flux')
 try:
  if not (run/'selected/render/render.json').exists():
   renv=dict(env,PATH='/mnt/ojc/miniconda3/envs/splatad/bin:/home/wmj/miniconda3/envs/sim/bin:'+env['PATH'],CUDA_HOME='/home/wmj/miniconda3/envs/sim',CPATH='/home/wmj/miniconda3/envs/splatad/include',TORCH_EXTENSIONS_DIR=str(W/'runs/cache/torch_extensions_sim_v2'))
   with (run/'render_selected.log').open('w') as log:
    rp=await asyncio.create_subprocess_exec('/home/wmj/miniconda3/envs/sim/bin/python',str(W/'src/render_selected.py'),name,env=renv,stdout=log,stderr=asyncio.subprocess.STDOUT)
    if await rp.wait():raise RuntimeError('Canonical render failed')
  with (run/('insert_'+mode+'.log')).open('w') as log:
   p=await asyncio.create_subprocess_exec(*cmd,env=env,cwd=str(R),stdout=log,stderr=asyncio.subprocess.STDOUT);code=await p.wait()
  (run/('insert_'+mode+'.exit')).write_text(str(code));print('INSERT_DONE',name,mode,code,flush=True)
 finally:pool.release(gpu_index)
async def main():
 name,mode=sys.argv[1:3];run=W/'runs'/name
 # Independent schedulers may request the same candidate; serialize per asset/mode.
 with (run/('insert_'+mode+'.lock')).open('a') as lease:
  while True:
   try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);break
   except BlockingIOError:await asyncio.sleep(2)
  try:
   flag=run/('insert_'+mode+'.exit');dest=run/('insertion_'+mode);mp=dest/'meta.json';traj=W/'configs'/('scene0103_'+mode+'.npz')
   if flag.exists() and flag.read_text().strip()=='0' and mp.exists() and traj.exists():
    meta=json.loads(mp.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    length=json.loads((W/'configs/metric_priors.json').read_text())[name]['length_m']
    if (meta.get('asset_sha256')==sha(run/'selected/canonical.ply') and meta.get('trajectory_sha256')==sha(traj)
        and meta.get('target_length_m')==length and meta.get('n_frames',0)>=24
        and (dest/'insert_splatad_6cam.mp4').is_file()):
     print('INSERT_CACHED',name,mode,flush=True);return
   await run_insertion()
  finally:fcntl.flock(lease,fcntl.LOCK_UN)
asyncio.run(main())
