import asyncio
import fcntl
import json
import os
import sys
from pathlib import Path
NEW=Path(__file__).resolve().parents[1];ROOT=NEW.parent;sys.path.insert(0,str(ROOT/'scripts'))
from baseline_queue import gpu_info,save,now


async def main():
    with (NEW/'logs/appearance_query.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=NEW/'ANALYSIS_STATUS.json';assert not path.exists();state=dict(state='running',jobs={},pid=os.getpid());save(path,state)
        async def job(label,index):
            record=dict(state='waiting_for_device');state['jobs'][label]=record;save(path,state)
            with (ROOT/'cache/gpu_locks'/f'gpu{index}.lock').open('a') as lock:
                while True:
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:await asyncio.sleep(10);continue
                    gpu=(await gpu_info())[index]
                    if gpu['free_mib']>=6000:break
                    fcntl.flock(lock,fcntl.LOCK_UN);await asyncio.sleep(15)
                with (NEW/'logs'/f'appearance_query_{label}.log').open('x') as log:
                    child=await asyncio.create_subprocess_exec(sys.executable,str(NEW/'scripts/appearance_query.py'),'--label',label,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid']),stdout=log,stderr=asyncio.subprocess.STDOUT)
                    record.update(state='running',pid=child.pid,device=gpu,started_at=now());save(path,state);code=await child.wait()
                record.update(state='completed_visual_review_pending' if code==0 else 'failed',exit_code=code,ended_at=now());save(path,state)
        await asyncio.gather(job('N04',4),job('N11',5))
        state.update(state='complete_visual_review_pending' if all(j['exit_code']==0 for j in state['jobs'].values()) else 'partial_requires_review');save(path,state)


if __name__=='__main__':asyncio.run(main())
