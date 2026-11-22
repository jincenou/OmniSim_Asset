"""Wait on the existing controller, evaluate paired temporal runs and render."""
import asyncio
import fcntl
import hashlib
import os
from pathlib import Path
from after_batch import ROUND,W,read,write,now
from resource_lease import lease,wait_memory

PY='/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python'

async def main():
    with (ROUND/'temporal_after.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=ROUND/'temporal_after.json'
        if path.exists():raise FileExistsError(path)
        state={'state':'waiting_for_training','pid':os.getpid(),'started_at':now(),'jobs':{}}
        def save(**fields):state.update(fields,updated_at=now());write(path,state)
        save()
        while True:
            upstream=read(ROUND/'temporal_queue.json')
            if upstream['state']=='all_controls_complete_review_pending':break
            if upstream['state']=='completed_with_failures':raise RuntimeError('Mobility training failed; preserve outputs')
            os.kill(upstream['pid'],0)
            await asyncio.sleep(30)
        labels=['walk063_rootlocal24','batch_0084_twentyfour'];recipes=['dense_relocate45k','dense_duration8'];jobs=[]
        for label in labels:
            data=(ROUND if label.startswith('walk063') else W)/'reconstruction/data'/label
            for recipe in recipes:
                checkpoint=ROUND/'quality'/f'{label}_{recipe}'/'ckpts/ckpt_49999.pt'
                jobs.append({'id':f'{label}_{recipe}','label':label,'recipe':recipe,'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'data':str(data)})
        manifest=ROUND/'benchmark/temporal_manifest.json'
        if manifest.exists():raise FileExistsError(manifest)
        write(manifest,{'state':'trained_before_evaluation','jobs':jobs,'plan':'../temporal_plan.json','protocol':'protocol.json','frozen_original_manifest':'manifest.json'})
        async def one(job,gpu):
            async with lease([gpu]) as fds:
                await wait_memory([gpu],6000)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TORCH_EXTENSIONS_DIR=str(W/'reconstruction/deps/torch_extensions'))
                env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
                cmd=[PY,str(ROUND/'scripts/benchmark_metrics.py'),'--job',job['id'],'--manifest',str(manifest)]
                log=ROUND/'logs'/f'temporal_metrics_{job["id"]}.log'
                with log.open('xb') as stream:
                    child=await asyncio.create_subprocess_exec(*cmd,env=env,stdout=stream,stderr=asyncio.subprocess.STDOUT,pass_fds=(*fds,own.fileno()))
                    row={'state':'evaluating','pid':child.pid,'gpu':gpu,'command':cmd,'log':str(log)};state['jobs'][job['id']]=row;save(state='evaluating');code=await child.wait()
                log.with_suffix('.exit').write_text(str(code)+'\n');row.update(state='complete' if code==0 else 'failed',exit_code=code);save()
                if code:raise RuntimeError(f'Metric evaluation failed: {job["id"]}')
        results=await asyncio.gather(*(one(job,gpu) for job,gpu in zip(jobs,[4,5,6,7],strict=True)),return_exceptions=True)
        errors=[str(x) for x in results if isinstance(x,BaseException)]
        if errors:save(state='evaluation_failed',failures=errors);raise RuntimeError('; '.join(errors))
        async def render(label,gpu):
            async with lease([gpu]) as fds:
                await wait_memory([gpu],6000)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',TORCH_EXTENSIONS_DIR=str(W/'reconstruction/deps/torch_extensions'))
                env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
                command=[PY,str(ROUND/'scripts/render_temporal_comparison.py'),'--label',label,'--mode','validation'];log=ROUND/'logs'/f'temporal_comparison_{label}.log'
                with log.open('xb') as stream:
                    child=await asyncio.create_subprocess_exec(*command,env=env,stdout=stream,stderr=asyncio.subprocess.STDOUT,pass_fds=(*fds,own.fileno()))
                    row={'state':'rendering','pid':child.pid,'gpu':gpu,'command':command,'log':str(log)};state['jobs'][label+'_video']=row;save(state='rendering');code=await child.wait()
                row.update(state='complete' if code==0 else 'failed',exit_code=code);log.with_suffix('.exit').write_text(str(code)+'\n');save()
                if code:raise RuntimeError(f'Video rendering failed: {label}')
        results=await asyncio.gather(render(labels[0],4),render(labels[1],5),return_exceptions=True)
        errors=[str(x) for x in results if isinstance(x,BaseException)]
        save(state='completed_with_failures' if errors else 'temporal_metrics_and_videos_complete_review_pending',failures=errors)
        if errors:raise RuntimeError('; '.join(errors))
        child=await asyncio.create_subprocess_exec(PY,str(ROUND/'scripts/temporal_report.py'))
        code=await child.wait()
        if code:save(state='report_failed',report_exit=code);raise RuntimeError('Temporal report failed')
        save(state='temporal_metrics_and_videos_complete_review_pending')

if __name__=='__main__':asyncio.run(main())
