import argparse
import asyncio
import fcntl
import json
import os
import sys
from pathlib import Path
NEW=Path(__file__).resolve().parents[1];ROOT=NEW.parent;sys.path.insert(0,str(ROOT/'scripts'))
from baseline_queue import gpu_info,save,now


async def main():
    parser=argparse.ArgumentParser();parser.add_argument('--trial',default='primary');args=parser.parse_args()
    with (NEW/'logs'/f'{args.trial}_evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=NEW/('EVALUATION_STATUS.json' if args.trial=='primary' else 'REPEAT_EVALUATION_STATUS.json');assert not path.exists();state=dict(state='waiting_for_replays',jobs={},pid=os.getpid(),trial=args.trial);save(path,state)
        async def worker(label,index):
            pending=set(['A','U','B'])
            while pending:
                ready=sorted(arm for arm in pending if (NEW/'runs'/args.trial/f'{label}_{arm}'/'validation/53000.json').exists())
                if not ready:
                    status=json.loads((NEW/('STATUS.json' if args.trial=='primary' else 'REPEAT_STATUS.json')).read_text())
                    failures={key.split('_')[1] for key,job in status['jobs'].items() if key.startswith(label+'_') and job['state'].startswith('failed')}
                    pending-=failures
                    if pending:await asyncio.sleep(15)
                    continue
                arm=ready[0];key=f'{label}_{arm}';record=dict(state='waiting_for_device');state['jobs'][key]=record;save(path,state)
                with (ROOT/'cache/gpu_locks'/f'gpu{index}.lock').open('a') as gpu_lock:
                    while True:
                        try:fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:await asyncio.sleep(15);continue
                        gpu=(await gpu_info())[index]
                        if gpu['free_mib']>=6000:break
                        fcntl.flock(gpu_lock,fcntl.LOCK_UN);await asyncio.sleep(15)
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'])
                    for script,phase in [('evaluate_boundary.py','global_and_regions'),('evaluate_parts.py','parts_and_time')]:
                        with (NEW/'logs'/f'{args.trial}_{key}_{phase}.log').open('x') as log:
                            child=await asyncio.create_subprocess_exec(sys.executable,str(NEW/'scripts'/script),'--label',label,'--arm',arm,'--trial',args.trial,env=env,stdout=log,stderr=asyncio.subprocess.STDOUT)
                            record.update(state=phase,pid=child.pid,device=gpu,started_at=now());save(path,state);code=await child.wait()
                        record.update(exit_code=code);save(path,state)
                        if code:break
                    record.update(state='metrics_complete_comparison_pending' if code==0 else 'failed',ended_at=now());save(path,state)
                pending.remove(arm)
        await asyncio.gather(worker('N04',6),worker('N11',7))
        state.update(state='metrics_complete_comparison_pending' if len(state['jobs'])==6 and all(j['state']=='metrics_complete_comparison_pending' for j in state['jobs'].values()) else 'partial_requires_review');save(path,state)


if __name__=='__main__':asyncio.run(main())
