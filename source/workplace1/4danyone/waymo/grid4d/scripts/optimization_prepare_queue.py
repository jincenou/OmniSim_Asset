import asyncio
import os
from baseline_queue import ROOT,save,now

async def main():
    path=ROOT/'optimization/PREPARE_STATUS.json';assert not path.exists()
    state=dict(state='running',jobs={},pid=os.getpid(),started_at=now());save(path,state)
    async def job(label):
        log=ROOT/'logs'/f'optimization_prepare_{label}.log'
        command=['/mnt/ojc/miniconda3/envs/grid4d/bin/python','-u',str(ROOT/'scripts/prepare_optimization.py'),'--label',label]
        with log.open('x') as out:
            child=await asyncio.create_subprocess_exec(*command,cwd=ROOT,stdout=out,stderr=asyncio.subprocess.STDOUT)
            record=dict(state='running',pid=child.pid,command=command,log=str(log),started_at=now());state['jobs'][label]=record;save(path,state);code=await child.wait()
        record.update(state='complete' if code==0 else 'failed',exit_code=code,ended_at=now());save(path,state)
    await asyncio.gather(job('N04'),job('N11'))
    state.update(state='complete' if all(j['state']=='complete' for j in state['jobs'].values()) else 'failed',ended_at=now());save(path,state)

if __name__=='__main__':asyncio.run(main())
