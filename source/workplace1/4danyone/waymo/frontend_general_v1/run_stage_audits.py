"""Bounded asynchronous CPU audits while eight GPUs run generation."""
from pathlib import Path
import asyncio,json,os,time
ROOT=Path(__file__).resolve().parent
PY='/mnt/ojc/miniconda3/envs/4danyone/bin/python'
CASES=('P18','P22','P41','P08','P36','P04','P30','P24')
state={'state':'running','jobs':{}}
def save():
    p=ROOT/'stage_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(p)
async def job(name,sem):
    if (ROOT/'stage_audit'/name/'audit.json').exists():
        state['jobs'][name]={'state':'complete_existing'};save();return
    async with sem:
        log=ROOT/'logs'/f'{name}_stages.log';log.parent.mkdir(exist_ok=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
          MPLCONFIGDIR=str(ROOT/'tmp/mpl'))
        with log.open('wb') as stream:
            process=await asyncio.create_subprocess_exec(PY,str(ROOT/'audit_motion_stages.py'),name,env=env,stdout=stream,stderr=asyncio.subprocess.STDOUT)
            state['jobs'][name]={'state':'running','pid':process.pid,'started':time.time()};save()
            code=await process.wait()
        state['jobs'][name].update(state='complete' if code==0 else 'failed',exit_code=code,ended=time.time());save()
async def main():
    sem=asyncio.Semaphore(2);await asyncio.gather(*(job(name,sem) for name in CASES))
    state['state']='finished_needs_review';save()
if __name__=='__main__':asyncio.run(main())
