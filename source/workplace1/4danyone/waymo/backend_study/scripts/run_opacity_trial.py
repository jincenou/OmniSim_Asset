"""Two bounded asynchronous diagnostic cases on the authorized free GPU."""
import asyncio,os,json,shutil,argparse
from run_queue import ROOT,ENV,query_gpus,save,stamp
DIAG=ROOT/'diagnostics/opacity'

async def case(label,gpu,evaluation_only=False):
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu['uuid'];env['PYTHONUNBUFFERED']='1'
    for condition in (['evaluate'] if evaluation_only else ['control','mask','evaluate']):
        assert shutil.disk_usage(ROOT).free>43*2**30
        log=DIAG/f'{label}_{condition}.log';receipt=log.with_suffix('.json')
        state=dict(label=label,condition=condition,gpu=gpu,started_at=stamp(),state='starting')
        with log.open('xb') as f:
            p=await asyncio.create_subprocess_exec(str(ENV),'-u',str(ROOT/'scripts/opacity_trial.py'),
                '--label',label,'--condition',condition,stdout=f,stderr=asyncio.subprocess.STDOUT,env=env,start_new_session=True)
            state.update(pid=p.pid,state='running');save(receipt,state)
            code=await p.wait();state.update(exit_code=code,ended_at=stamp(),state='complete' if code==0 else 'failed');save(receipt,state)
        assert code==0,f'{label}/{condition} failed; inspect {log}'
    return dict(label=label,state='evaluated_visual_review_pending')

async def main(evaluation_only=False):
    gpu=next(g for g in await query_gpus() if g['physical_gpu']==7)
    assert gpu['free_mib']>16000
    state=dict(state='running',started_at=stamp(),pid=os.getpid(),gpu=gpu,evaluation_only=evaluation_only)
    save(DIAG/'STATUS.json',state)
    values=await asyncio.gather(case('N04',gpu,evaluation_only),case('N08',gpu,evaluation_only),return_exceptions=True)
    state.update(state='failed_partial' if any(isinstance(v,Exception) for v in values) else 'evaluated_visual_review_pending',
        results=[str(v) if isinstance(v,Exception) else v for v in values],ended_at=stamp())
    save(DIAG/'STATUS.json',state)
    assert state['state']!='failed_partial',state

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--evaluation-only',action='store_true')
    asyncio.run(main(parser.parse_args().evaluation_only))
