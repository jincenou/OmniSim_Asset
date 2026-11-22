import asyncio,json,os,sys
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'DriveLab'))
from asset_extraction.generation import GpuPool
async def main():
 spec=json.loads(Path(sys.argv[1]).read_text());pool=GpuPool(spec.get('gpus',[1]),timeout_seconds=300)
 gpu=await pool.acquire(spec.get('stage','frontend'))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1');env.update(spec.get('env',{}))
 try:
  with open(spec['log'],'a') as log:
   p=await asyncio.create_subprocess_exec(*spec['command'],cwd=spec.get('cwd',str(W/'DriveLab')),env=env,stdout=log,stderr=asyncio.subprocess.STDOUT);code=await p.wait()
  Path(spec['log']+'.exit').write_text(str(code));print('EXIT',code,'log',spec['log']);return code
 finally:pool.release(gpu)
sys.exit(asyncio.run(main()))
