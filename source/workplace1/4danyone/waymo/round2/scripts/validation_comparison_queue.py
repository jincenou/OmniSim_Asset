"""Wait for the live long-budget scheduler; render comparisons on leased6/7."""
import asyncio
import fcntl
import os
from after_batch import ROUND,W,read,write,now
from resource_lease import lease,wait_memory

async def main():
    with (ROUND/'validation_comparison.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        statepath=ROUND/'validation_comparison.json'
        if statepath.exists():raise FileExistsError(statepath)
        state={'state':'waiting_for50k','pid':os.getpid(),'started_at':now(),'jobs':{}}
        def save(**fields):state.update(fields,updated_at=now());write(statepath,state)
        save()
        while True:
            upstream=read(ROUND/'independent_validation_queue.json')
            if upstream['state']=='four_independent_validation_controls_evaluated':break
            if upstream['state']=='completed_with_failures':raise RuntimeError('Upstream50k failed; no comparison produced')
            # Terminal controller absence is a failure, not permission to restart it.
            os.kill(upstream['pid'],0)
            await asyncio.sleep(30)
        async def render(label,gpu):
            async with lease([gpu]) as fds:
                await wait_memory([gpu],6000)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    CUDA_HOME='/mnt/ojc/miniconda3/envs/asset_harvester_2026',
                    TORCH_EXTENSIONS_DIR=str(W/'reconstruction/deps/torch_extensions'),MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.9',OMP_NUM_THREADS='4')
                env['PATH']=env['CUDA_HOME']+'/bin:'+env['PATH'];env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
                command=['/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python',str(ROUND/'scripts/render_budget_comparison.py'),'--label',label,'--mode','validation']
                log=ROUND/'logs'/f'validation_comparison_{label}.log'
                with log.open('xb') as stream:
                    child=await asyncio.create_subprocess_exec(*command,env=env,stdout=stream,stderr=asyncio.subprocess.STDOUT,pass_fds=(*fds,own.fileno()))
                    record={'state':'running','pid':child.pid,'gpu':gpu,'command':command,'log':str(log),'started_at':now()}
                    state['jobs'][label]=record;save(state='rendering');code=await child.wait()
                log.with_suffix('.exit').write_text(str(code)+'\n');record.update(exit_code=code,ended_at=now(),state='failed' if code else 'rendered_decoded_visual_review_pending');save()
                if code:raise RuntimeError(f'{label} rendering failed')
        results=await asyncio.gather(render('batch_0073_twentyfour',6),render('batch_0086_twentyfour',7),return_exceptions=True)
        failures=[str(x) for x in results if isinstance(x,BaseException)]
        save(state='completed_with_failures' if failures else 'two_validation_comparisons_complete',failures=failures)
        if failures:raise RuntimeError('; '.join(failures))


if __name__=='__main__':asyncio.run(main())
