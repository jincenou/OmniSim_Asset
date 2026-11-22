"""After paired control success, render synchronized61-frame comparison in tmux."""
import asyncio
import argparse
import os
from pathlib import Path
from after_batch import ROUND,W,read,write,now
from resource_lease import lease,wait_memory

async def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sigma8',action='store_true');args=parser.parse_args()
    suffix='_sigma8' if args.sigma8 else ''
    path=ROUND/('weak_velocity_after'+suffix+'.json')
    if path.exists():raise FileExistsError(path)
    write(path,{'state':'waiting_for_pair','pid':os.getpid(),'started_at':now()})
    while True:
        state=read(ROUND/('weak_velocity_duration_queue.json' if args.sigma8 else 'weak_velocity_queue.json'))['state']
        if state=='failed':raise RuntimeError('Paired training failed; no comparison')
        if state=='pair_trained_evaluated_review_pending':break
        await asyncio.sleep(30)
    async with lease([4]) as handles:
        await wait_memory([4],6000)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',
            CUDA_HOME='/mnt/ojc/miniconda3/envs/asset_harvester_2026',TORCH_EXTENSIONS_DIR=str(W/'reconstruction/deps/torch_extensions'),
            TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        env['PATH']=env['CUDA_HOME']+'/bin:'+env['PATH'];env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        command=['/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python',str(ROUND/'scripts/render_weak_velocity.py')]+(['--sigma8'] if args.sigma8 else [])
        with (ROUND/'logs'/('weak_velocity_comparison'+suffix+'.log')).open('xb') as log:
            process=await asyncio.create_subprocess_exec(*command,cwd=ROUND/'backend',env=env,stdout=log,stderr=asyncio.subprocess.STDOUT,pass_fds=handles)
            write(path,{'state':'rendering','pid':os.getpid(),'child':process.pid,'command':command,'started_at':now()})
            code=await process.wait()
        (ROUND/'logs'/('weak_velocity_comparison'+suffix+'.exit')).write_text(str(code)+'\n')
        write(path,{'state':'comparison_complete_visual_review_pending' if code==0 else 'failed','exit_code':code,'ended_at':now()})
        if code:raise SystemExit(code)

if __name__=='__main__':asyncio.run(main())
