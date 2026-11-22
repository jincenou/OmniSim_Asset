"""Validate reviewed candidates in the real scene; archive completed candidates, never auto-accept."""
import asyncio,json,sys
from pathlib import Path
W=Path(__file__).resolve().parents[1];P='/mnt/ojc/miniconda3/envs/vllm_vlm/bin/python'
async def group(names,gpu):
 for name in names:
  run=W/'runs'/name
  assert json.loads((run/'selected/load_validation.json').read_text())['passed']
  support=json.loads((run/'selected/support_validation.json').read_text())
  if support['support_difference_m']>.04:print('HOLD_SUPPORT',name,flush=True);continue
  with (run/'insert_dispatch.log').open('w') as log:
   p=await asyncio.create_subprocess_exec(P,str(W/'src/insert_demo.py'),name,'near_follow',str(gpu),stdout=log,stderr=asyncio.subprocess.STDOUT)
   code=await p.wait()
  flag=run/'insert_near_follow.exit'
  if code or not flag.exists() or flag.read_text().strip()!='0':print('INSERT_FAILED',name,flush=True);continue
  with (run/'package.log').open('w') as log:
   p=await asyncio.create_subprocess_exec(P,str(W/'src/package_candidate.py'),name,'near_follow',stdout=log,stderr=asyncio.subprocess.STDOUT)
   code=await p.wait()
  print('PACKAGE_CANDIDATE',name,code,flush=True)
async def main():
 names=[r['name'] for r in json.loads((W/'sources/web_expansion_20261103/queue.json').read_text())]
 await asyncio.gather(group(names[::2],1),group(names[1::2],6))
 print('BATCH_INSERT_FINISHED_REQUIRES_VISUAL_REVIEW',flush=True)
asyncio.run(main())
