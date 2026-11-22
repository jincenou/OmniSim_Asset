"""Async CPU validation after observed generator process completion."""
import asyncio,json,os,time
from pathlib import Path
import run_base_ablation as base
ROOT=base.ROOT;STATE={'state':'waiting_generators','jobs':{}};CPU=asyncio.Semaphore(2)
def save():base.dump(ROOT/'publish_status.json',STATE)
async def watch(case):
 name=case.split('/')[-1]
 while True:
  status=json.loads((ROOT/'base_resume_status.json').read_text())['jobs'].get(name)
  if status is None:raise RuntimeError(f'Missing recovery job {name}')
  if status['state']=='failed':STATE['jobs'][name]={'state':'generation_failed','detail':status};save();return
  if status['state']=='generated_needs_validation':break
  if status['state']!='running':raise RuntimeError(status)
  # A stale JSON state alone is not evidence of a running job.
  try:os.kill(status['pid'],0)
  except ProcessLookupError:
   await asyncio.sleep(2)
   fresh=json.loads((ROOT/'base_resume_status.json').read_text())['jobs'][name]
   if fresh==status:raise RuntimeError(f'Generator PID missing and state not finalized: {name}')
   continue
  STATE['jobs'][name]={'state':'waiting_live_generator','pid':status['pid'],'checked_at':time.time()};save();await asyncio.sleep(30)
 async with CPU:
  with (ROOT/'logs'/f'{name}_publish.log').open('x') as log:
   p=await asyncio.create_subprocess_exec(base.PY,str(ROOT/'publish_base.py'),case,stdout=log,stderr=log)
   STATE['jobs'][name]={'state':'publishing','pid':p.pid};save();rc=await p.wait()
   STATE['jobs'][name]={'state':'published_needs_quality_review' if rc==0 else 'publish_failed','exit_code':rc};save()
async def protected(case):
 try:await watch(case)
 except Exception as e:STATE['jobs'][case.split('/')[-1]]={'state':'failed','error':repr(e)};save()
async def main():
 if (ROOT/'publish_status.json').exists():raise FileExistsError('Inspect existing publisher before recovery')
 save();await asyncio.gather(*(protected(case) for case in base.CASES));STATE['state']='queue_finished';save()
if __name__=='__main__':asyncio.run(main())
