from pathlib import Path
import asyncio,json,os,time,hashlib
ROOT=Path(__file__).resolve().parent;PY='/mnt/ojc/miniconda3/envs/4danyone/bin/python'
state={'state':'running','jobs':{}}
def save():
    p=ROOT/'candidate_v2_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(p)
async def job(name,variant,sem):
    key=name+'/'+variant
    async with sem:
        state['jobs'][key]={'state':'running','started':time.time()};save()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLCONFIGDIR=str(ROOT/'tmp/mpl'))
        async def run(script):
            with (ROOT/'logs'/f'{name}_{variant}_{script}.log').open('wb') as log:
                p=await asyncio.create_subprocess_exec(PY,str(ROOT/(script+'.py')),name,'--variant',variant,env=env,stdout=log,stderr=asyncio.subprocess.STDOUT)
                state['jobs'][key].update(stage=script,pid=p.pid);save();return await p.wait()
        code=await run('optimize_motion')
        if code==0:code=await run('review_motion')
        state['jobs'][key].update(state='ready_for_review' if code==0 else 'failed',exit_code=code,ended=time.time());save()
async def main():
    cfg=json.loads((ROOT/'protocol_v2.json').read_text());sem=asyncio.Semaphore(2)
    record={'frozen_utc_seconds':time.time(),'scope':'Recorded before V2 pilot optimization launch','files':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['optimize_motion.py','review_motion.py','gap_proposal.py','camera_geometry.py','protocol_v2.json']}}
    p=ROOT/'implementation_v2_freeze.json'
    with p.open('x') as f:json.dump(record,f,indent=2)
    await asyncio.gather(*(job(name,variant,sem) for name in cfg['pilot_subjects'] for variant in cfg['branches']))
    state['state']='finished_candidates_not_accepted';save()
if __name__=='__main__':asyncio.run(main())
