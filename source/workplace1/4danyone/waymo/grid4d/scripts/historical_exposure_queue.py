import asyncio
import json
import os
from baseline_queue import ROOT,save,now


async def main():
    path=ROOT/'audit/historical_exposure_queue.json';assert not path.exists();sem=asyncio.Semaphore(2)
    state=dict(state='running',pid=os.getpid(),started_at=now(),jobs={});save(path,state)
    async def job(label):
        async with sem:
            log=ROOT/'logs'/f'historical_exposure_{label}.log'
            with log.open('x') as out:
                process=await asyncio.create_subprocess_exec('/mnt/ojc/miniconda3/envs/grid4d/bin/python','-u',str(ROOT/'scripts/historical_exposure.py'),'--label',label,stdout=out,stderr=asyncio.subprocess.STDOUT)
                record=dict(state='running',pid=process.pid,log=str(log));state['jobs'][label]=record;save(path,state);code=await process.wait()
            record.update(state='complete' if code==0 else 'failed',exit_code=code);save(path,state)
    await asyncio.gather(*(job(label) for label in ['N04','N11','N01','N07','N02','N03','N06','N08']))
    state.update(state='complete' if all(j['state']=='complete' for j in state['jobs'].values()) else 'partial_requires_review',ended_at=now());save(path,state)


if __name__=='__main__':asyncio.run(main())
