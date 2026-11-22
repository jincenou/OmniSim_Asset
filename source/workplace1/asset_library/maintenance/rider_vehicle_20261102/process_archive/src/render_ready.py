import sys,asyncio,os
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'src/DriveLab'))
from asset_extraction.generation import GpuPool
from asset_extraction.io_utils import read_json
async def main():
 run=Path(sys.argv[1]);gpu_index=int(sys.argv[2]) if len(sys.argv)>2 else 1;pool=GpuPool([gpu_index]);env=dict(os.environ,PATH='/mnt/ojc/miniconda3/envs/splatad/bin:/home/wmj/miniconda3/envs/sim/bin:/usr/local/cuda/bin:'+os.environ['PATH'],CUDA_HOME='/home/wmj/miniconda3/envs/sim',CPATH='/home/wmj/miniconda3/envs/splatad/include',CUDA_VISIBLE_DEVICES=str(gpu_index),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMBA_CACHE_DIR=str(W/'runs/cache/numba'),TORCH_EXTENSIONS_DIR=str(W/'runs/cache/torch_extensions_sim_v2'),MAX_JOBS='4',PYTHONUNBUFFERED='1')
 for method in (sys.argv[3:] or ['H0']):
  work=run/method
  if not work.exists():continue
  for _ in range(240):
   r=read_json(work/'generation.json')['assets'][0]
   if r['status'] in ['generated','failed']:break
   await asyncio.sleep(5)
  if r['status']!='generated':print('RENDER_SKIPPED',method,r['status'],flush=True);continue
  gpu=await pool.acquire('render')
  try:
   with (work/'render.log').open('w') as log:
    p=await asyncio.create_subprocess_exec('/home/wmj/miniconda3/envs/sim/bin/python',str(W/'src/render_methods.py'),str(run),'--method',method,env=env,stdout=log,stderr=asyncio.subprocess.STDOUT)
    code=await p.wait();print(method,'render exit',code,flush=True)
  finally:pool.release(gpu)
asyncio.run(main())
