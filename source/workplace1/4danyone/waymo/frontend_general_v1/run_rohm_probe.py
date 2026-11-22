from pathlib import Path
import asyncio,json,os,time,hashlib
ROOT=Path(__file__).resolve().parent;PY='/mnt/ojc/miniconda3/envs/4danyone/bin/python'
UUID='GPU-0db9aa59-0293-a3ea-06a0-395913880fbd'
state={'state':'running','jobs':{}}
def save():
    p=ROOT/'rohm_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(p)
async def process(name,script,env):
    with (ROOT/'logs'/f'{name}_{script}.log').open('wb') as log:
        p=await asyncio.create_subprocess_exec(PY,str(ROOT/(script+'.py')),name,env=env,stdout=log,stderr=asyncio.subprocess.STDOUT)
        state['jobs'][name].update(stage=script,pid=p.pid);save();return await p.wait()
async def run(name,cpu_sem,gpu_sem):
    state['jobs'][name]={'state':'preparing','started':time.time()};save()
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLCONFIGDIR=str(ROOT/'tmp/mpl'))
    code=0
    if not (ROOT/'rohm_proposals'/name/'preparation.json').exists():
        async with cpu_sem:code=await process(name,'prepare_rohm',dict(env,CUDA_VISIBLE_DEVICES=''))
    if code==0:
        async with gpu_sem:
            # Shared GPU permitted; avoid starting even the small prior if headroom disappears.
            while True:
                p=await asyncio.create_subprocess_exec('nvidia-smi','--id='+UUID,'--query-gpu=memory.free','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE)
                output,_=await p.communicate()
                if p.returncode!=0:raise RuntimeError('Cannot validate GPU headroom')
                if int(output.strip())>=4096:break
                state['jobs'][name]['state']='waiting_4g_headroom';save();await asyncio.sleep(30)
            state['jobs'][name]['state']='sampling';save()
            code=await process(name,'sample_rohm',dict(env,CUDA_VISIBLE_DEVICES=UUID,FD_EXPECTED_GPU_UUID=UUID))
    state['jobs'][name].update(state='sampled_needs_alignment' if code==0 else 'failed',exit_code=code,ended=time.time());save()
async def main():
    config=json.loads((ROOT/'rohm_protocol_v1.json').read_text());cpu=asyncio.Semaphore(2);gpu=asyncio.Semaphore(1)
    record={'frozen_utc_seconds':time.time(),'before_first_real_sampling':True,'files':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['prepare_rohm.py','sample_rohm.py','rohm_protocol_v1.json','run_rohm_probe.py']},'joint_mask_detail':'Positions use named COCO observations; limb rotation conditioning additionally requires its mapped child evidence; only untrusted eight limb rotations may be applied, torso/head/hand rotations preserved.'}
    with (ROOT/'rohm_implementation_freeze.json').open('x') as f:json.dump(record,f,indent=2)
    await asyncio.gather(*(run(name,cpu,gpu) for name in config['subjects']))
    state['state']='finished_needs_alignment_review';save()
if __name__=='__main__':asyncio.run(main())
