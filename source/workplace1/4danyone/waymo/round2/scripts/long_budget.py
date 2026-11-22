"""Two 50k-budget controls; preserve all previous outputs and per-GPU leases."""
import asyncio
import fcntl
import os
import shutil
from after_batch import ROUND,W,read,write,now
from resource_lease import lease,wait_memory

PY='/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python'
STATE=ROUND/'long_budget.json'


async def main():
    with (ROUND/'long_budget.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if STATE.exists():raise FileExistsError(STATE)
        state={'state':'running','pid':os.getpid(),'started_at':now(),'jobs':{},
            'purpose':'50k total optimization and LR horizon; same Q0 inputs, initialization, loss and fixed relocation cutoff4500. Not an exact paper implementation.',
            'paper_reference':'https://arxiv.org/html/2608.20335#A5',
            'baseline_batch_limit':12}
        def save(**fields):
            state.update(fields,updated_at=now());write(STATE,state)
        save()
        async def identity(label,gpu):
            async with lease([gpu]) as fds:
                await wait_memory([gpu],12000)
                if shutil.disk_usage(ROUND).free<30*1024**3:raise RuntimeError('Disk reserve below30GiB')
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    CUDA_HOME='/mnt/ojc/miniconda3/envs/asset_harvester_2026',
                    TORCH_EXTENSIONS_DIR=str(W/'reconstruction/deps/torch_extensions'),MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.9')
                env['PATH']=env['CUDA_HOME']+'/bin:'+env['PATH'];env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
                name=f'{label}_Q50';out=ROUND/'quality'/name;log=ROUND/'logs'/f'{name}.log'
                if out.exists():raise FileExistsError(out)
                command=[PY,str(ROUND/'scripts/train_control.py'),'--label',label,'--views','21','--steps','50000',
                    '--recipe','budget_half_50k','--output',str(out)]
                with log.open('xb') as handle:
                    process=await asyncio.create_subprocess_exec(*command,cwd=ROUND/'backend',env=env,
                        stdout=handle,stderr=asyncio.subprocess.STDOUT,pass_fds=(*fds,own.fileno()))
                    record={'state':'running','pid':process.pid,'gpu':gpu,'command':command,'log':str(log),'started_at':now()}
                    state['jobs'][name]=record;save()
                    code=await process.wait();log.with_suffix('.exit').write_text(str(code)+'\n')
                record.update(exit_code=code,ended_at=now())
                if code:
                    record.update(state='failed');save();raise RuntimeError(f'{name} exited {code}')
                if read(out/'quality_metrics/metrics.json')['steps']!=50000:raise RuntimeError('Missing50k metric evidence')
                record.update(state='trained_and_evaluated_visual_review_pending');save()
        results=await asyncio.gather(identity('phone_full',4),identity('pilot_043_twentyfour',5),return_exceptions=True)
        failures=[str(value) for value in results if isinstance(value,BaseException)]
        save(state='completed_with_failures' if failures else 'two50k_controls_evaluated',failures=failures)
        if failures:raise RuntimeError('; '.join(failures))


if __name__=='__main__':asyncio.run(main())
