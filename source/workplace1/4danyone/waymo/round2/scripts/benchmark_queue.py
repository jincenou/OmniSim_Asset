"""Bounded async workers on authorized shared GPUs; immutable result folders."""
import asyncio
import fcntl
import os
from after_batch import ROUND,read,write,now
from resource_lease import lease,wait_memory

async def main():
    root=ROUND/'benchmark';path=root/'queue.json'
    with (root/'queue.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if path.exists():raise FileExistsError(path)
        jobs=asyncio.Queue()
        for job in read(root/'manifest.json')['jobs']:jobs.put_nowait(job)
        state={'state':'running','pid':os.getpid(),'started_at':now(),'jobs':{},'shared_gpus':[4,5,6,7]}
        def save():state['updated_at']=now();write(path,state)
        save()
        async def worker(gpu):
            async with lease([gpu]) as handles:
                while not jobs.empty():
                    job=jobs.get_nowait();name=job['id'];await wait_memory([gpu],6000)
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TORCH_EXTENSIONS_DIR=str(ROUND.parent/'reconstruction/deps/torch_extensions'))
                    env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
                    cmd=['/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python',str(ROUND/'scripts/benchmark_metrics.py'),'--job',name]
                    log=root/'logs'/f'{name}.log'
                    with log.open('xb') as stream:
                        child=await asyncio.create_subprocess_exec(*cmd,env=env,stdout=stream,stderr=asyncio.subprocess.STDOUT,pass_fds=(*handles,own.fileno()))
                        row={'state':'running','gpu':gpu,'pid':child.pid,'command':cmd,'log':str(log),'started_at':now()};state['jobs'][name]=row;save()
                        code=await child.wait()
                    row.update(state='complete' if code==0 else 'failed',exit_code=code,ended_at=now());save();jobs.task_done()
        await asyncio.gather(worker(4),worker(5),worker(6),worker(7))
        state['state']='complete' if all(x['exit_code']==0 for x in state['jobs'].values()) else 'completed_with_failures';save()

if __name__=='__main__':asyncio.run(main())
