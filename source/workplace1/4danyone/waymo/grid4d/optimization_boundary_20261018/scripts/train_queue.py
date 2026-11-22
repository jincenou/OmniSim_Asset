"""Bounded async GPU jobs, shared project device locks, explicit failure receipts."""
import argparse
import asyncio
import fcntl
import json
import os
import shutil
import sys
from pathlib import Path
NEW=Path(__file__).resolve().parents[1];ROOT=NEW.parent
sys.path.insert(0,str(ROOT/'scripts'))
from baseline_queue import gpu_info,save,now


async def main():
    parser=argparse.ArgumentParser();parser.add_argument('--trial',default='primary',choices=['primary','repeat']);args=parser.parse_args()
    with (NEW/'logs'/f'{args.trial}_queue.lock').open('a') as main_lock:
        fcntl.flock(main_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=NEW/('STATUS.json' if args.trial=='primary' else 'REPEAT_STATUS.json')
        if path.exists():assert not json.loads(path.read_text()).get('jobs'),'Inspect existing queue before starting again'
        state=dict(state='running',trial=args.trial,pid=os.getpid(),started_at=now(),jobs={});save(path,state)
        async def job(label,arm,index):
            key=f'{label}_{arm}';record=dict(state='waiting_for_device',gpu_index=index);state['jobs'][key]=record;save(path,state)
            with (ROOT/'cache/gpu_locks'/f'gpu{index}.lock').open('a') as lock:
                while True:
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:await asyncio.sleep(15);continue
                    gpu=(await gpu_info())[index]
                    if gpu['free_mib']>=16000 and shutil.disk_usage(NEW).free>35*2**30:break
                    fcntl.flock(lock,fcntl.LOCK_UN);record.update(state='waiting_for_memory',gpu=gpu);save(path,state);await asyncio.sleep(20)
                environment=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid']);environment.pop('PYTHONPATH',None)
                async def execute(command,name):
                    logfile=NEW/'logs'/f'{args.trial}_{key}_{name}.log'
                    with logfile.open('x') as output:
                        child=await asyncio.create_subprocess_exec(*command,cwd=NEW,env=environment,stdout=output,stderr=asyncio.subprocess.STDOUT)
                        record.update(state=name,pid=child.pid,gpu=gpu,command=command,log=str(logfile),phase_started_at=now());save(path,state);code=await child.wait()
                    record.update(exit_code=code,phase_ended_at=now());save(path,state);return code
                command=[sys.executable,'-u',str(NEW/'scripts/train_boundary.py'),'--label',label,'--arm',arm,'--trial',args.trial]
                code=await execute(command,'training')
                if code:record.update(state='failed_training');save(path,state);return
                run=NEW/'runs'/args.trial/key
                for iteration in [49000,51000,53000]:
                    code=await execute([sys.executable,str(NEW/'scripts/validate_boundary.py'),'--run',str(run),'--iteration',str(iteration)],f'replay_{iteration}')
                    if code:record.update(state='failed_replay',iteration=iteration);save(path,state);return
                record.update(state='trained_replayed_evaluation_pending',ended_at=now());save(path,state)
        await asyncio.gather(job('N04','A',6),job('N11','A',7),job('N04','U',0),job('N11','U',1),job('N04','B',2),job('N11','B',3))
        state.update(state='training_replay_complete_evaluation_pending' if all(v['state']=='trained_replayed_evaluation_pending' for v in state['jobs'].values()) else 'partial_requires_review',ended_at=now());save(path,state)


if __name__=='__main__':asyncio.run(main())
