"""Bounded independent GPU lanes; one object's complete pipeline per lane."""
import asyncio,os,json,time,shutil,fcntl
from pathlib import Path
N=Path(__file__).resolve().parents[1];LOCKS=N.parent.parent/'gpu_leases_20261019';LOCKS.mkdir(exist_ok=True)
def save(p,x):
 q=p.with_suffix('.writing');q.write_text(json.dumps(x,ensure_ascii=False,indent=2));q.replace(p)
async def main():
 plan=json.loads((N/'PLAN.json').read_text());assert plan['state']=='frozen_before_training';p=await asyncio.create_subprocess_exec('nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE);out,_=await p.communicate();assert p.returncode==0;gpus=[dict(index=int(i),uuid=u.strip(),free=int(m)) for i,u,m in (s.split(',') for s in out.decode().strip().splitlines())];jobs=[f'{l}_{a}' for l in ['N06','N11'] for a in ['G0','G1','G2']];states={};history=[]
 def status():save(N/'STATUS.json',dict(state='running' if any(r.get('state')=='running' for r in states.values()) else 'queue_pending_or_terminal',at=time.time(),jobs=states,completed=sum(r.get('state')=='complete' for r in states.values()),total=6,frontend_parallel='separate scene20 pipeline shares leases; no other tasks terminated'))
 async def run(key,gpu):
  label=key[:3];arm=key[-2:];lease=(LOCKS/f'gpu{gpu["index"]}.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);assert gpu['free']>12000;env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu['uuid'];env['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
  stages=[('train',['train.py','--label',label,'--arm',arm],N/'runs'/key/'training.complete.json'),('export',['export.py','--key',key],N/'assets'/key/'export_validation.json'),('replay',['assets.py','--asset',str(N/'assets'/key),'--expected',str(N/'runs'/key/'replay_expected.npz')],N/'assets'/key/'replay_validation.json'),('evaluate',['evaluate.py','--key',key,'--step','32999'],N/'results'/f'{key}_32999/metrics.json'),('parts',['evaluate_parts.py','--label',label,'--arm',f'{key}_32999'],N/'results'/f'{key}_32999/parts.json'),('expanded',['evaluate_expanded.py','--label',label,'--key',f'{key}_32999'],N/'results'/f'{key}_32999/expanded.json')]
  for name,args,marker in stages:
   if marker.exists():continue
   assert shutil.disk_usage(N).free>5*2**30
   with (N/'logs'/f'{key}_{name}.log').open('a') as log:
    started=time.time();proc=await asyncio.create_subprocess_exec('/mnt/ojc/miniconda3/envs/grid4d/bin/python',str(N/'scripts'/args[0]),*args[1:],env=env,stdout=log,stderr=log);states[key]=dict(state='running',stage=name,pid=proc.pid,gpu=gpu,started=started);status();rc=await proc.wait();receipt=dict(key=key,stage=name,pid=proc.pid,exit_code=rc,seconds=time.time()-started,gpu=gpu);history.append(receipt);save(N/'queue_history.json',history)
    if rc!=0 or not marker.exists():states[key].update(state='failed',exit_code=rc);status();lease.close();return
  states[key].update(state='complete',ended=time.time());status();lease.close()
 await asyncio.gather(*(run(key,gpu) for key,gpu in zip(jobs,gpus[:6])))
 save(N/'pipeline_queue.complete.json',dict(all_passed=all(r.get('state')=='complete' for r in states.values()),jobs=states));status()
asyncio.run(main())
