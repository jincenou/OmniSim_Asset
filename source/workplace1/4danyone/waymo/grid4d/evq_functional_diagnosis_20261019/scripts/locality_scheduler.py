import asyncio,os,json,time
from pathlib import Path
D=Path(__file__).resolve().parents[1]
async def main():
 jobs=[('N06_R2','GPU-cd4b2342-fba4-2b42-beb9-60b698debc2d'),('N06_R3','GPU-32040e1a-df6d-0776-5420-fae47bc11248'),('N11_R2','GPU-0db9aa59-0293-a3ea-06a0-395913880fbd'),('N11_R3','GPU-4e635188-f5de-33f9-6ecb-53555a0a7c9d')];status={}
 async def run(key,uuid):
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=uuid
  with (D/'logs'/f'{key}_locality.log').open('a') as h:
   p=await asyncio.create_subprocess_exec('/mnt/ojc/miniconda3/envs/grid4d/bin/python',str(D/'scripts/locality.py'),'--key',key,stdout=h,stderr=h,env=env);status[key]=dict(pid=p.pid,uuid=uuid,started=time.time());rc=await p.wait();status[key].update(exit_code=rc,ended=time.time())
 await asyncio.gather(*(run(k,u) for k,u in jobs));(D/'research/locality_jobs.json').write_text(json.dumps(status,indent=2))
asyncio.run(main())
