"""Read-only start and snapshot probes on two separate shared GPU workers."""
import asyncio
import fcntl
import json
import os
import sys
from pathlib import Path
NEW=Path(__file__).resolve().parents[1];ROOT=NEW.parent;sys.path.insert(0,str(ROOT/'scripts'))
from baseline_queue import gpu_info,save,now


async def main():
    with (NEW/'logs/diagnostics.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=NEW/'DIAGNOSTIC_STATUS.json';assert not path.exists();state=dict(state='running',jobs={},pid=os.getpid());save(path,state)
        async def execute(label,index,run=None,iteration=48000):
            key=label+'_start' if run is None else run.parent.name+'_'+run.name+f'_{iteration}'
            record=dict(state='waiting',gpu_index=index);state['jobs'][key]=record;save(path,state)
            with (ROOT/'cache/gpu_locks'/f'gpu{index}.lock').open('a') as gpu_lock:
                while True:
                    try:fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:await asyncio.sleep(15);continue
                    gpu=(await gpu_info())[index]
                    if gpu['free_mib']>=6000:break
                    fcntl.flock(gpu_lock,fcntl.LOCK_UN);await asyncio.sleep(15)
                command=[sys.executable,str(NEW/'scripts/gradient_probe.py'),'--label',label]
                if run is not None:command+=['--run',str(run),'--iteration',str(iteration)]
                with (NEW/'logs'/f'grad_{key}.log').open('x') as log:
                    child=await asyncio.create_subprocess_exec(*command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid']),stdout=log,stderr=asyncio.subprocess.STDOUT)
                    record.update(state='running',pid=child.pid,started_at=now());save(path,state);code=await child.wait()
                record.update(state='passed' if code==0 else 'failed',exit_code=code,ended_at=now());save(path,state)
        async def worker(label,index):
            await execute(label,index)
            pending={(arm,iteration) for arm in ['A','U','B'] for iteration in [49000,51000,53000]}
            while pending:
                ready=sorted([(arm,it) for arm,it in pending if (NEW/'runs/primary'/f'{label}_{arm}'/'snapshots'/f'{it}.json').exists()],key=lambda x:(x[1],x[0]))
                if ready:
                    arm,it=ready[0];await execute(label,index,NEW/'runs/primary'/f'{label}_{arm}',it);pending.remove((arm,it));continue
                queue=json.loads((NEW/'STATUS.json').read_text())
                failed={key.split('_')[1] for key,entry in queue['jobs'].items() if key.startswith(label+'_') and entry['state'].startswith('failed')}
                pending={(arm,it) for arm,it in pending if arm not in failed}
                if pending:await asyncio.sleep(15)
        await asyncio.gather(worker('N04',4),worker('N11',5))
        state.update(state='complete' if all(j['state']=='passed' for j in state['jobs'].values()) else 'partial_requires_review',ended_at=now());save(path,state)


if __name__=='__main__':asyncio.run(main())
