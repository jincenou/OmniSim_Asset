"""Use released owned GPU slots for full-timeline asset-only replay, no optimization."""
import asyncio,fcntl,json,os,sys,time
from pathlib import Path
NEW=Path(__file__).resolve().parents[1];ROOT=NEW.parent
from run_cross8 import UUIDS
PLAN=json.loads((NEW/'PLAN.json').read_text());status=dict(state='running',pid=os.getpid(),jobs={});active=set()
def save():
    p=NEW/'PROFILE_STATUS.json';q=p.with_suffix('.writing');q.write_text(json.dumps(status,indent=2)+'\n');q.replace(p)
async def one(name,it,gpu,lock):
    e=status['jobs'][name]=dict(state='running',gpu=gpu,started_at=time.time());save()
    try:
        with (NEW/'logs'/f'profile_{name}.log').open('x') as log:
            p=await asyncio.create_subprocess_exec(sys.executable,str(NEW/'scripts/profile_replay.py'),'--run',str(NEW/'runs'/name),'--iteration',str(it),env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUIDS[gpu]),stdout=log,stderr=asyncio.subprocess.STDOUT);e['pid']=p.pid;save();code=await p.wait()
        e.update(state='complete' if code==0 else 'failed',exit_code=code,ended_at=time.time());save()
    finally:lock.close();active.remove(gpu)
async def main():
    pending=[(l+'_'+s,it) for l in PLAN['new_A_labels'] for s,it in [('33k',33000),('alpha',48000),('A',53000)]]+[(l+'_fused',33000) for l in PLAN['new_fused_labels']];tasks=[];save()
    while pending:
        controller=json.loads((NEW/'STATUS.json').read_text());jobs=controller['jobs']
        # Never race the controller's two unassigned jobs for its nonblocking GPU locks.
        if any(j['state']=='queued' for j in jobs.values()):await asyncio.sleep(25);continue
        busy={j['gpu'] for j in jobs.values() if j['state'] not in ['complete','failed'] and 'gpu' in j}
        for name,it in pending.copy():
            if len(active)>=2:break
            run=NEW/'runs'/name
            if not (run/'complete.json').exists():
                if controller['state']!='running':status['jobs'][name]=dict(state='not_produced');save();pending.remove((name,it))
                continue
            free=[g for g in [6,7,0,1,2,3,4,5] if g not in busy|active]
            if not free:break
            gpu=free[0];lock=(ROOT/'cache/gpu_locks'/f'gpu{gpu}.lock').open('a')
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:lock.close();continue
            active.add(gpu);tasks.append(asyncio.create_task(one(name,it,gpu,lock)));pending.remove((name,it))
        if pending:await asyncio.sleep(25)
    await asyncio.gather(*tasks);status['state']='complete' if all(j['state']=='complete' for j in status['jobs'].values()) else 'terminal_with_failures';save()
if __name__=='__main__':asyncio.run(main())
