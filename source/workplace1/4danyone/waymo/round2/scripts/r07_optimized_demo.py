"""Standalone viewing package for an existing diagnostic checkpoint; no training."""
import asyncio
import os
from after_batch import ROUND,W,write,now
from resource_lease import lease,wait_memory

async def main():
    state=ROUND/'r07_optimized_demo.json'
    if state.exists():raise FileExistsError(state)
    write(state,{'state':'waiting_for_gpu','pid':os.getpid(),'gpu':7,'started_at':now()})
    async with lease([7]) as handles:
        await wait_memory([7],6000)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',CUDA_DEVICE_ORDER='PCI_BUS_ID',
            CUDA_HOME='/mnt/ojc/miniconda3/envs/asset_harvester_2026',TORCH_EXTENSIONS_DIR=str(W/'reconstruction/deps/torch_extensions'),
            TORCH_CUDA_ARCH_LIST='8.9',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
        env['PATH']=env['CUDA_HOME']+'/bin:'+env['PATH'];env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        cmd=['/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python',str(ROUND/'scripts/render_demo61.py'),'--job',str(ROUND/'demo_jobs/r07_short61_zero_sigma8.json')]
        with (ROUND/'logs/r07_optimized_demo.log').open('xb') as log:
            p=await asyncio.create_subprocess_exec(*cmd,cwd=ROUND/'backend',env=env,stdout=log,stderr=asyncio.subprocess.STDOUT,pass_fds=handles)
            write(state,{'state':'rendering','pid':os.getpid(),'child':p.pid,'command':cmd,'started_at':now()})
            code=await p.wait()
        (ROUND/'logs/r07_optimized_demo.exit').write_text(str(code)+'\n')
        write(state,{'state':'rendered_and_decoded_visual_review_pending' if code==0 else 'failed','exit_code':code,'ended_at':now()})
        if code:raise SystemExit(code)

if __name__=='__main__':asyncio.run(main())
