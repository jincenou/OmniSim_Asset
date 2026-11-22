import argparse
import asyncio
import fcntl
import json
import os
from baseline_queue import ROOT,gpu_info,save,now


async def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=int,default=1);args=parser.parse_args()
    path=ROOT/'mechanism/static/STATUS.json';assert not path.exists()
    state=dict(state='running',jobs={},pid=os.getpid(),started_at=now());save(path,state)
    async def job(label,frame,index):
        with (ROOT/'cache/gpu_locks'/f'gpu{index}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            gpu=(await gpu_info())[index];assert gpu['free_mib']>6000
            key=f'{label}_t{frame:03d}';log=ROOT/'logs'/f'static_{key}_attempt{args.attempt}.log'
            command=['/mnt/ojc/miniconda3/envs/grid4d/bin/python','-u',str(ROOT/'scripts/static_diagnostic.py'),'--label',label,'--frame',str(frame)]
            with log.open('x') as out:
                proc=await asyncio.create_subprocess_exec(*command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid']),stdout=out,stderr=asyncio.subprocess.STDOUT)
                record=dict(state='running',pid=proc.pid,gpu=gpu,command=command,log=str(log),started_at=now());state['jobs'][key]=record;save(path,state);code=await proc.wait()
            record.update(state='complete' if code==0 else 'failed',exit_code=code,ended_at=now());save(path,state)
    await asyncio.gather(job('N04',0,6),job('N04',60,2),job('N11',0,7),job('N11',60,3))
    state.update(state='training_complete_evaluation_pending' if all(v['state']=='complete' for v in state['jobs'].values()) else 'partial_requires_review',ended_at=now());save(path,state)


if __name__=='__main__':asyncio.run(main())
