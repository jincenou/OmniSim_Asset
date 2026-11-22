import asyncio,os,sys
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'src/DriveLab'))
from asset_extraction.generation import GpuPool
async def main():
 pool=GpuPool([int(sys.argv[2])]);gpu=await pool.acquire('render')
 try:
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
  p=await asyncio.create_subprocess_exec('/mnt/ojc/miniconda3/envs/4danyone-sam2/bin/python',str(W/'src/segment_packet.py'),sys.argv[1],env=env)
  if await p.wait():raise RuntimeError('SAM2 failed')
 finally:pool.release(gpu)
asyncio.run(main())
