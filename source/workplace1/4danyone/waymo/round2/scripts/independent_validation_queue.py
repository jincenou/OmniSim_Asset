"""Four independent controlled jobs on4/5/6/7, coordinated with existing leases."""
import asyncio
import fcntl
import os
import shutil
from pathlib import Path
from dataclasses import asdict
from after_batch import ROUND,W,read,write,now
from resource_lease import lease,wait_memory

PY='/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python'


async def main():
    with (ROUND/'independent_validation_queue.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=ROUND/'independent_validation_queue.json'
        if path.exists():raise FileExistsError(path)
        if read(ROUND/'optimization_queue.json')['state']!='four_optimization_controls_evaluated':raise RuntimeError('Tuning controls have not completed')
        state={'state':'running','pid':os.getpid(),'started_at':now(),'jobs':{},'baseline_batch_limit':12,
            'hypothesis':'Frozen higher-density recipe showed better aggregate LPIPS and outside-alpha on two tuning tracks; verify on two additional tracks against their own50k baseline.'}
        def save(**fields):state.update(fields,updated_at=now());write(path,state)
        save()
        async def one(label,recipe,gpu):
            name=f'{label}_{recipe}';out=ROUND/'quality'/name
            env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_HOME='/mnt/ojc/miniconda3/envs/asset_harvester_2026',
                TORCH_EXTENSIONS_DIR=str(W/'reconstruction/deps/torch_extensions'),MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.9',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
            env['PATH']=env['CUDA_HOME']+'/bin:'+env['PATH'];env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
            record={'state':'preparing','gpu':gpu,'recipe':recipe,'label':label,'started_at':now()};state['jobs'][name]=record;save()
            if recipe=='capacity50k':
                prep=[PY,str(ROUND/'scripts/dense_initialization.py'),'--label',label];log=ROUND/'logs'/f'{name}_init.log'
                with log.open('xb') as stream:
                    child=await asyncio.create_subprocess_exec(*prep,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=stream,stderr=asyncio.subprocess.STDOUT)
                    record.update(state='preparing_dense_init_cpu',preparation={'pid':child.pid,'command':prep,'log':str(log)});save()
                    code=await child.wait()
                log.with_suffix('.exit').write_text(str(code)+'\n');record['preparation']['exit_code']=code;save()
                if code:record.update(state='preparation_failed');save();raise RuntimeError(f'{name} initialization failed')
            # Inspect the actual Config before allocating a GPU.
            from train_control import configuration
            original=asdict(configuration(label,21,50000,out,recipe='budget_half_50k')[0])
            cfg=asdict(configuration(label,21,50000,out,recipe=recipe)[0])
            delta={key:[original[key],cfg[key]] for key in original if original[key]!=cfg[key]}
            expected=set() if recipe=='budget_half_50k' else {'init_npz_path'}
            if set(delta)!=expected:raise ValueError(f'Unexpected configuration difference:{delta}')
            record.update(state='waiting_for_gpu_lease',configuration_delta=delta);save()
            async with lease([gpu]) as fds:
                await wait_memory([gpu],6000)
                if shutil.disk_usage(ROUND).free<30*1024**3:raise RuntimeError('Disk reserve below30GiB')
                env['CUDA_VISIBLE_DEVICES']=str(gpu)
                command=[PY,str(ROUND/'scripts/train_control.py'),'--label',label,'--views','21','--steps','50000','--recipe',recipe,'--output',str(out)]
                log=ROUND/'logs'/f'{name}.log'
                with log.open('xb') as stream:
                    child=await asyncio.create_subprocess_exec(*command,env=env,stdout=stream,stderr=asyncio.subprocess.STDOUT,pass_fds=(*fds,own.fileno()))
                    record.update(state='training',pid=child.pid,command=command,log=str(log));save();code=await child.wait()
                log.with_suffix('.exit').write_text(str(code)+'\n');record.update(exit_code=code,ended_at=now(),state='failed' if code else 'trained_evaluated_visual_review_pending');save()
                if code:raise RuntimeError(f'{name} failed; checkpoint and logs retained')
                metrics=read(out/'quality_metrics/metrics.json')
                if metrics['steps']!=50000:raise RuntimeError('Incomplete50k output')
        results=await asyncio.gather(one('batch_0073_twentyfour','budget_half_50k',4),one('batch_0086_twentyfour','budget_half_50k',5),
            one('batch_0073_twentyfour','capacity50k',6),one('batch_0086_twentyfour','capacity50k',7),return_exceptions=True)
        failures=[str(value) for value in results if isinstance(value,BaseException)]
        save(state='completed_with_failures' if failures else 'four_independent_validation_controls_evaluated',failures=failures)
        if failures:raise RuntimeError('; '.join(failures))


if __name__=='__main__':asyncio.run(main())
