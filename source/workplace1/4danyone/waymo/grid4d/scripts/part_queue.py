import asyncio
import json
import os
from baseline_queue import ROOT,save,now


async def main():
    path=ROOT/'audit/part_queue.json';assert not path.exists()
    assert json.loads((ROOT/'research/temporal_control.json').read_text())['state']=='passed'
    state=dict(state='running',jobs={},pid=os.getpid(),started_at=now());sem=asyncio.Semaphore(4)
    async def job(label,method):
        async with sem:
            while not (ROOT/'audit/roi'/label/'manifest.json').exists():
                q=json.loads((ROOT/'audit/diagnostic_queue.json').read_text())
                if q['jobs'].get(label,{}).get('state')=='failed':raise RuntimeError(label+' preparation failed')
                await asyncio.sleep(10)
            key=label+'_'+method;log=ROOT/'logs'/f'part_{key}.log'
            with log.open('x') as out:
                command=['/mnt/ojc/miniconda3/envs/4dgaussians/bin/python','-u',str(ROOT/'scripts/part_metrics.py'),'--label',label,'--method',method]
                p=await asyncio.create_subprocess_exec(*command,cwd=ROOT,stdout=out,stderr=asyncio.subprocess.STDOUT)
                rec=dict(state='running',pid=p.pid,command=command,log=str(log));state['jobs'][key]=rec;save(path,state);code=await p.wait()
            rec.update(state='complete' if code==0 else 'failed',exit_code=code);save(path,state)
    await asyncio.gather(*(job(label,method) for label in ['N04','N11','N01','N07','N02','N03','N06','N08'] for method in ['ftgs_base5k','ftgs_opt15k','4dgs_grid25','4dgs_grid61']))
    state.update(state='complete' if all(v['state']=='complete' for v in state['jobs'].values()) else 'partial_requires_review',ended_at=now());save(path,state)


if __name__=='__main__':asyncio.run(main())
