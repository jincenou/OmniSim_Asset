"""Finish baseline12 diagnostics and compare Q0/Q1/Q2 on identical native targets."""
import asyncio
import fcntl
import os
from pathlib import Path
import shutil
from after_batch import ROUND,W,read,write,now
from resource_lease import lease,wait_memory

PY='/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python'


async def main():
    with (ROUND/'delivery_metrics.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        statepath=ROUND/'delivery_metrics.json'
        if statepath.exists():raise FileExistsError(statepath)
        jobs=[]
        for item in read(W/'batch100/closeout12.json')['assets']:
            label=item['label']
            if label in ('phone_full','pilot_043_twentyfour'):
                existing=read(ROUND/'quality'/f'baseline_{label}'/'quality_metrics/metrics.json')
                if existing['steps']!=5000:raise RuntimeError('Existing baseline evidence has wrong step')
                continue
            jobs.append((f'baseline_{label}',W/'reconstruction/outputs'/label/'ckpts/ckpt_4999.pt',W/'reconstruction/data'/label))
        for label in ('phone_full','pilot_043_twentyfour'):
            native=ROUND/'quality_masters'/label/'native'
            if read(native/'manifest.json')['state']!='native_rgb_with_fixed_alpha_master_ready':raise RuntimeError('Native master incomplete')
            jobs.extend([(f'{label}_Q0_native_eval',W/'reconstruction/outputs'/label/'ckpts/ckpt_4999.pt',native),
                (f'{label}_Q1_native_eval',ROUND/'quality'/f'{label}_Q1'/'ckpts/ckpt_14999.pt',native)])
        state={'state':'running','pid':os.getpid(),'started_at':now(),'jobs':{},'planned_jobs':len(jobs)}
        def save(**fields):state.update(fields,updated_at=now());write(statepath,state)
        save()
        pending=asyncio.Queue()
        for job in jobs:pending.put_nowait(job)
        async def worker(gpu):
            async with lease([gpu]) as fds:
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    CUDA_HOME='/mnt/ojc/miniconda3/envs/asset_harvester_2026',
                    TORCH_EXTENSIONS_DIR=str(W/'reconstruction/deps/torch_extensions'),MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.9',OMP_NUM_THREADS='4')
                env['PATH']=env['CUDA_HOME']+'/bin:'+env['PATH'];env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
                while not pending.empty():
                    name,checkpoint,data=pending.get_nowait()
                    await wait_memory([gpu],10000)
                    if shutil.disk_usage(ROUND).free<30*1024**3:raise RuntimeError('Disk reserve below30GiB')
                    output=ROUND/'quality'/name/'quality_metrics';log=ROUND/'logs'/f'{name}_delivery_metrics.log'
                    command=[PY,str(ROUND/'scripts/evaluate_asset.py'),'--checkpoint',str(checkpoint),'--data',str(data),'--output',str(output)]
                    with log.open('xb') as stream:
                        child=await asyncio.create_subprocess_exec(*command,env=env,stdout=stream,stderr=asyncio.subprocess.STDOUT,pass_fds=(*fds,lock.fileno()))
                        entry={'state':'running','pid':child.pid,'command':command,'gpu':gpu,'log':str(log),'started_at':now()}
                        state['jobs'][name]=entry;save();code=await child.wait()
                    log.with_suffix('.exit').write_text(str(code)+'\n')
                    entry.update(exit_code=code,ended_at=now(),state='failed' if code else 'evaluated_visual_review_pending');save()
                    if code:raise RuntimeError(f'{name} failed; see preserved log')
        results=await asyncio.gather(worker(6),worker(7),return_exceptions=True)
        failures=[str(x) for x in results if isinstance(x,BaseException)]
        save(state='completed_with_failures' if failures else 'fourteen_evaluations_complete',failures=failures)
        if failures:raise RuntimeError('; '.join(failures))


if __name__=='__main__':asyncio.run(main())
