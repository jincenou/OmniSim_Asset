import asyncio
import os
from baseline_queue import ROOT,save,now


async def main():
    path=ROOT/'audit/diagnostic_queue.json';assert not path.exists()
    state=dict(state='running',jobs={},pid=os.getpid(),started_at=now());sem=asyncio.Semaphore(4)
    async def sample(label):
        async with sem:
            log=ROOT/'logs'/f'diagnostic_prepare_{label}.log'
            with log.open('x') as out:
                command=['/mnt/ojc/miniconda3/envs/4dgaussians/bin/python','-u',str(ROOT/'scripts/prepare_diagnostics.py'),'--label',label]
                child=await asyncio.create_subprocess_exec(*command,cwd=ROOT,stdout=out,stderr=asyncio.subprocess.STDOUT)
                rec=dict(state='running',pid=child.pid,command=command,log=str(log),started_at=now());state['jobs'][label]=rec;save(path,state)
                code=await child.wait()
            rec.update(state='complete' if code==0 else 'failed',exit_code=code,ended_at=now());save(path,state)
    await asyncio.gather(*(sample(label) for label in ['N04','N11','N01','N07','N02','N03','N06','N08']))
    state.update(state='complete' if all(v['state']=='complete' for v in state['jobs'].values()) else 'partial_requires_review',ended_at=now());save(path,state)


if __name__=='__main__':asyncio.run(main())
