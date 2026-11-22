import asyncio,json,sys
from pathlib import Path
W=Path(__file__).resolve().parents[1];P='/mnt/ojc/miniconda3/envs/vllm_vlm/bin/python'
async def group(names,gpu):
 for name in names:
  work=W/'runs'/name/'H0'
  with (work/'render_dispatch.log').open('w') as log:
   p=await asyncio.create_subprocess_exec(P,str(W/'src/render_ready.py'),str(work.parent),str(gpu),'H0',stdout=log,stderr=asyncio.subprocess.STDOUT)
   code=await p.wait()
  print('ORBIT_CONTROLLER',name,code,flush=True)
async def main():
 names=[r['name'] for r in json.loads((W/'sources/web_expansion_20261103/queue.json').read_text()) if r['status']=='ready_for_H']
 await asyncio.gather(group(names[::2],1),group(names[1::2],6))
asyncio.run(main())
