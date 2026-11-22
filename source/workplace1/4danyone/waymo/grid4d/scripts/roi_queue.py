import asyncio
import os
from baseline_queue import ROOT,save,now


async def main():
    state=dict(state='running',jobs={},started_at=now(),pid=os.getpid());sem=asyncio.Semaphore(4)
    async def case(label):
        async with sem:
            log=ROOT/'logs'/f'roi_{label}.log'
            with log.open('x') as output:
                proc=await asyncio.create_subprocess_exec('/mnt/ojc/miniconda3/envs/4dgaussians/bin/python','-u',str(ROOT/'scripts/roi_candidates.py'),'--label',label,cwd=ROOT,stdout=output,stderr=asyncio.subprocess.STDOUT)
                rec=dict(state='running',pid=proc.pid,log=str(log));state['jobs'][label]=rec;save(ROOT/'audit/roi_queue.json',state)
                code=await proc.wait()
            rec.update(exit_code=code,state='complete' if code==0 else 'failed');save(ROOT/'audit/roi_queue.json',state)
    await asyncio.gather(*(case(label) for label in ['N04','N11','N01','N07','N02','N03','N06','N08']))
    state.update(state='candidates_ready_for_review' if all(v['state']=='complete' for v in state['jobs'].values()) else 'failed_requires_review',ended_at=now())
    save(ROOT/'audit/roi_queue.json',state)


if __name__=='__main__':asyncio.run(main())
