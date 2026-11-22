"""Apply the frozen O2 losses to the O3 proposal, reuse exact duplicate starts."""
from pathlib import Path
import asyncio,json,os,time
import torch
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parent;PY='/mnt/ojc/miniconda3/envs/4danyone/bin/python'
state={'state':'running','jobs':{}}
def save():
    p=ROOT/'rohm_alignment_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(p)
async def run(name,sem):
    async with sem:
        state['jobs'][name]={'state':'checking','started':time.time()};save()
        prior=ROOT/'rohm_proposals'/name
        assert (prior/'sample_report.json').exists()
        proposed=load_file(str(prior/'proposal.safetensors'))['gap_proposal_25fps']
        existing=ROOT/'motion_candidates_v2'/name/'local_temporal'
        old=load_file(str(existing/'motion_candidates.safetensors'))
        key='proposal_initial.body_pose' if 'proposal_initial.body_pose' in old else 'proposal_O1.body_pose'
        if torch.equal(proposed,old[key]):
            state['jobs'][name].update(state='exact_initial_proposal_reused',result=str(existing),evidence='torch.equal on entire float32 initial pose; same fixed camera, shape, observations and V2 loss protocol',ended=time.time());save();return
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLCONFIGDIR=str(ROOT/'tmp/mpl'))
        async def execute(script):
            with (ROOT/'logs'/f'{name}_rohm_{script}.log').open('wb') as log:
                p=await asyncio.create_subprocess_exec(PY,str(ROOT/(script+'.py')),name,'--variant','rohm',env=env,stdout=log,stderr=asyncio.subprocess.STDOUT)
                state['jobs'][name].update(state='running',stage=script,pid=p.pid);save();return await p.wait()
        code=await execute('optimize_motion')
        if code==0:code=await execute('review_motion')
        state['jobs'][name].update(state='ready_for_review' if code==0 else 'failed',exit_code=code,result=str(ROOT/'motion_candidates_v2'/name/'rohm'),ended=time.time());save()
async def main():
    sem=asyncio.Semaphore(2);cfg=json.loads((ROOT/'rohm_alignment_protocol.json').read_text())
    await asyncio.gather(*(run(name,sem) for name in cfg['subjects']))
    state['state']='finished_not_quality_approved';save()
if __name__=='__main__':asyncio.run(main())
