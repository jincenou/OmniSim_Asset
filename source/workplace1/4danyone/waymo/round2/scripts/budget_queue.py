"""Q1 then Q2 for each identity; GPUs4/5 leave6/7 for walking reconstruction."""
import asyncio
import fcntl
import os
import shutil
from after_batch import ROUND,W,read,write,now
from resource_lease import lease,wait_memory

PY='/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python'
STATE=ROUND/'budget_queue.json'


async def main():
    with (ROUND/'budget_queue.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if STATE.exists():raise FileExistsError(STATE)
        state={'state':'waiting_for_walk24_and_native_masters','pid':os.getpid(),'started_at':now(),'jobs':{}}
        def save(**fields):state.update(fields,updated_at=now());write(STATE,state)
        save()
        try:
            while True:
                walk=read(ROUND/'walk24_run.json');native_exit=ROUND/'logs/native_master.exit'
                if native_exit.exists() and native_exit.read_text().strip()!='0':raise RuntimeError('Native master preparation failed')
                if walk['state'] in ('24_view_generated_visual_review_pending','failed') and native_exit.exists():break
                await asyncio.sleep(60)
            save(state='running')
            async def identity(label,gpu):
                async with lease([gpu]) as fds:
                    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',
                        CUDA_HOME='/mnt/ojc/miniconda3/envs/asset_harvester_2026',
                        TORCH_EXTENSIONS_DIR=str(W/'reconstruction/deps/torch_extensions'),MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.9')
                    env['PATH']=env['CUDA_HOME']+'/bin:'+env['PATH'];env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
                    for recipe,tag in [('budget_half','Q1'),('budget_native','Q2')]:
                        await wait_memory([gpu],12000)
                        if shutil.disk_usage(ROUND).free<30*1024**3:raise RuntimeError('Disk reserve below30GiB')
                        name=f'{label}_{tag}';out=ROUND/'quality'/name;log=ROUND/'logs'/f'{name}.log'
                        if out.exists():raise FileExistsError(out)
                        command=[PY,str(ROUND/'scripts/train_control.py'),'--label',label,'--views','21','--steps','15000',
                            '--recipe',recipe,'--output',str(out)]
                        with log.open('xb') as handle:
                            process=await asyncio.create_subprocess_exec(*command,cwd=ROUND/'backend',env=env,
                                stdout=handle,stderr=asyncio.subprocess.STDOUT,pass_fds=(*fds,own.fileno()))
                            record={'state':'running','pid':process.pid,'gpu':gpu,'command':command,'log':str(log),'started_at':now()}
                            state['jobs'][name]=record;save()
                            code=await process.wait();log.with_suffix('.exit').write_text(str(code)+'\n')
                        record.update(exit_code=code,ended_at=now())
                        if code:record.update(state='failed');save();raise RuntimeError(f'{name} exited {code}')
                        metrics=read(out/'quality_metrics/metrics.json')
                        if metrics['steps']!=15000:raise RuntimeError('Missing final metric evidence')
                        record.update(state='trained_and_evaluated_visual_review_pending');save()
            results=await asyncio.gather(identity('phone_full',4),identity('pilot_043_twentyfour',5),return_exceptions=True)
            failures=[str(value) for value in results if isinstance(value,BaseException)]
            save(state='completed_with_failures' if failures else 'four_budget_runs_evaluated',failures=failures)
        except Exception as error:save(state='failed',error=f'{type(error).__name__}: {error}');raise


if __name__=='__main__':asyncio.run(main())
