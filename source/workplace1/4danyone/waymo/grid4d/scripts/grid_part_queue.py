import asyncio
from baseline_queue import ROOT,save,now


async def main():
    state=dict(state='running',jobs={},started_at=now());path=ROOT/'audit/grid_part_queue.json';assert not path.exists()
    async def case(label):
        log=ROOT/'logs'/f'part_{label}_grid4d_33k.log'
        with log.open('x') as out:
            p=await asyncio.create_subprocess_exec('/mnt/ojc/miniconda3/envs/4dgaussians/bin/python','-u',str(ROOT/'scripts/part_metrics.py'),
                '--label',label,'--method','grid4d_33k',cwd=ROOT,stdout=out,stderr=asyncio.subprocess.STDOUT)
            rec=dict(state='running',pid=p.pid,log=str(log));state['jobs'][label]=rec;save(path,state);code=await p.wait()
        rec.update(state='complete' if code==0 else 'failed',exit_code=code);save(path,state)
    await asyncio.gather(case('N04'),case('N11'))
    state.update(state='complete' if all(v['state']=='complete' for v in state['jobs'].values()) else 'partial_requires_review',ended_at=now());save(path,state)


if __name__=='__main__':asyncio.run(main())
