from pathlib import Path
import asyncio,json,os,time
ROOT=Path(__file__).resolve().parent;PY='/mnt/ojc/miniconda3/envs/4danyone/bin/python'
state={'state':'running','jobs':{}}
def save():
    p=ROOT/'candidate_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(p)
async def job(name,sem):
    async with sem:
        state['jobs'][name]={'state':'running','started':time.time()};save()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLCONFIGDIR=str(ROOT/'tmp/mpl'))
        async def run(script):
            with (ROOT/'logs'/f'{name}_{script}.log').open('wb') as log:
                p=await asyncio.create_subprocess_exec(PY,str(ROOT/(script+'.py')),name,env=env,stdout=log,stderr=asyncio.subprocess.STDOUT)
                state['jobs'][name].update(stage=script,pid=p.pid);save();return await p.wait()
        code=0
        if not (ROOT/'motion_candidates'/name/'report.json').exists():code=await run('optimize_motion')
        if code==0:code=await run('review_motion')
        state['jobs'][name].update(state='ready_for_review' if code==0 else 'failed',exit_code=code,ended=time.time());save()
async def main():
    cohort=json.loads((ROOT/'cohort_v1.json').read_text());sem=asyncio.Semaphore(2)
    await asyncio.gather(*(job(x['sample_id'],sem) for x in cohort['dev6']+cohort['auxiliary_seen']))
    state['state']='finished_candidates_not_accepted';save()
if __name__=='__main__':asyncio.run(main())
