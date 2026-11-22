"""Render baseline demos once; leases coordinate with training/evaluation."""
import asyncio
import fcntl
import os
import shutil
from after_batch import ROUND,W,read,write,now
from resource_lease import lease,wait_memory

PY='/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python'


async def main():
    with (ROUND/'demo_queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=ROUND/'demo_queue.json'
        if path.exists():raise FileExistsError(path)
        state={'state':'waiting_for_gpu_leases','pid':os.getpid(),'started_at':now(),'jobs':{}}
        def save(**fields):state.update(fields,updated_at=now());write(path,state)
        save()
        async def run(label,gpu):
            async with lease([gpu]) as fds:
                await wait_memory([gpu],10000)
                if shutil.disk_usage(ROUND).free<30*1024**3:raise RuntimeError('Disk reserve below30GiB')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    CUDA_HOME='/mnt/ojc/miniconda3/envs/asset_harvester_2026',
                    TORCH_EXTENSIONS_DIR=str(W/'reconstruction/deps/torch_extensions'),MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.9',OMP_NUM_THREADS='4')
                env['PATH']=env['CUDA_HOME']+'/bin:'+env['PATH'];env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
                command=[PY,str(ROUND/'scripts/render_demo.py'),'--job',str(ROUND/'demo_jobs'/f'{label}.json')]
                log=ROUND/'logs'/f'demo_{label}.log'
                with log.open('xb') as stream:
                    child=await asyncio.create_subprocess_exec(*command,env=env,stdout=stream,stderr=asyncio.subprocess.STDOUT,pass_fds=(*fds,lock.fileno()))
                    record={'state':'running','pid':child.pid,'gpu':gpu,'command':command,'log':str(log),'started_at':now()}
                    state['jobs'][label]=record;save(state='rendering');code=await child.wait()
                log.with_suffix('.exit').write_text(str(code)+'\n')
                record.update(exit_code=code,ended_at=now(),state='failed' if code else 'rendered_decoded_visual_review_pending');save()
                if code:raise RuntimeError(f'{label} failed, see log')
                if read(ROUND/'demos'/label/'validation.json')['state']!='rendered_decoded_visual_review_pending':raise RuntimeError('Missing validation')
        # Exercise one complete real asset before expanding this new renderer.
        await run('phone_full',6)
        queue=asyncio.Queue()
        for a in read(W/'batch100/closeout12.json')['assets']:
            if a['label']!='phone_full':queue.put_nowait(a['label'])
        async def worker(gpu):
            while not queue.empty():await run(queue.get_nowait(),gpu)
        results=await asyncio.gather(worker(6),worker(7),return_exceptions=True)
        failures=[str(x) for x in results if isinstance(x,BaseException)]
        save(state='completed_with_failures' if failures else 'twelve_demo_packages_rendered_and_decoded',failures=failures)
        if failures:raise RuntimeError('; '.join(failures))


if __name__=='__main__':asyncio.run(main())
