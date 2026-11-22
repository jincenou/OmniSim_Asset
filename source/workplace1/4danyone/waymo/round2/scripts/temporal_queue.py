"""Four bounded independent controls, respecting existing per-GPU leases."""
import asyncio
import fcntl
import os
import shutil
from after_batch import ROUND,W,read,write,now
from resource_lease import lease,wait_memory
PY='/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python'

async def main():
    with (ROUND/'temporal_controls.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=ROUND/'temporal_queue.json'
        if path.exists():raise FileExistsError(path)
        state={'state':'running','pid':os.getpid(),'started_at':now(),'jobs':{}}
        def save():state.update(updated_at=now());write(path,state)
        save()
        async def case(job):
            name=job['id'];record={'state':'waiting','gpu':job['gpu']};state['jobs'][name]=record;save()
            try:
                async with lease([job['gpu']]) as handles:
                    await wait_memory([job['gpu']],10000)
                    if shutil.disk_usage(ROUND).free<35*1024**3:raise RuntimeError('Disk reserve below35GiB')
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(job['gpu']),CUDA_DEVICE_ORDER='PCI_BUS_ID',
                        CUDA_HOME='/mnt/ojc/miniconda3/envs/asset_harvester_2026',TORCH_EXTENSIONS_DIR=str(W/'reconstruction/deps/torch_extensions'),
                        TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
                    env['PATH']=env['CUDA_HOME']+'/bin:'+env['PATH'];env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
                    cmd=[PY,str(ROUND/'scripts/temporal_control.py'),'--label',job['label'],'--variant',job['variant']]
                    log=ROUND/'logs'/f'temporal_{name}.log'
                    with log.open('xb') as stream:
                        p=await asyncio.create_subprocess_exec(*cmd,cwd=ROUND/'backend',env=env,stdout=stream,
                            stderr=asyncio.subprocess.STDOUT,pass_fds=(*handles,own.fileno()))
                        record.update(state='running',pid=p.pid,command=cmd,log=str(log),started_at=now());save()
                        code=await p.wait()
                    log.with_suffix('.exit').write_text(str(code)+'\n');record.update(exit_code=code,ended_at=now())
                    if code:raise RuntimeError(f'Exit {code}: {log}')
                    record['state']='trained_evaluated_visual_review_pending'
            except Exception as error:record.update(state='failed',error=f'{type(error).__name__}: {error}')
            finally:save()
        await asyncio.gather(*map(case,read(ROUND/'temporal_plan.json')['jobs']))
        state['state']='completed_with_failures' if any(x['state']=='failed' for x in state['jobs'].values()) else 'all_controls_complete_review_pending'
        save()
        if state['state']=='completed_with_failures':raise SystemExit(1)

if __name__=='__main__':asyncio.run(main())
