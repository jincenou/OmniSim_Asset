import asyncio
import os
from after_batch import ROUND, W, write, now
from resource_lease import lease, wait_memory

async def main():
    state = ROUND/'r07_beforeafter_job.json'
    if state.exists():
        raise FileExistsError(state)
    write(state, {'state':'waiting_for_gpu6', 'started_at':now(), 'pid':os.getpid()})
    async with lease([6]) as handles:
        await wait_memory([6], 6000)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='6', CUDA_DEVICE_ORDER='PCI_BUS_ID',
            CUDA_HOME='/mnt/ojc/miniconda3/envs/asset_harvester_2026',
            TORCH_EXTENSIONS_DIR=str(W/'reconstruction/deps/torch_extensions'),
            OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        env['PATH'] = env['CUDA_HOME']+'/bin:'+env['PATH']
        env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
        command = ['/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python',str(ROUND/'scripts/render_r07_beforeafter.py')]
        with (ROUND/'logs/r07_beforeafter.log').open('xb') as log:
            child = await asyncio.create_subprocess_exec(*command,cwd=ROUND/'backend',env=env,
                stdout=log,stderr=asyncio.subprocess.STDOUT,pass_fds=handles)
            write(state, {'state':'rendering','pid':os.getpid(),'child_pid':child.pid,'gpu':6,'command':command,'started_at':now()})
            code = await child.wait()
        (ROUND/'logs/r07_beforeafter.exit').write_text(str(code)+'\n')
        write(state, {'state':'rendered_decoded_review_pending' if code==0 else 'failed','exit_code':code,'ended_at':now()})
        if code:
            raise SystemExit(code)

if __name__ == '__main__':
    asyncio.run(main())
