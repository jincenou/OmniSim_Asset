import asyncio,os,json
from pathlib import Path
D=Path(__file__).resolve().parents[1]
async def main():
 sem=asyncio.Semaphore(3);results={}
 async def job(key):
  if (D/"comparison"/key/"media.json").exists():
   results[key]=0;return
  async with sem:
   with (D/'logs'/f'{key}_media.log').open('w') as h:
    p=await asyncio.create_subprocess_exec('/mnt/ojc/miniconda3/envs/grid4d/bin/python',str(D/'scripts/media.py'),'--key',key,stdout=h,stderr=h);rc=await p.wait();results[key]=rc
 await asyncio.gather(*(job(n+'_'+r) for n in ['N06','N11'] for r in ['R1','R2','R3']));(D/'research/media_jobs.json').write_text(json.dumps(results,indent=2))
asyncio.run(main())
