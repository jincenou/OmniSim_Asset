"""Background target segmentation and three motion conditions, after leases free."""
import asyncio
import fcntl
import os
import shutil
from after_batch import ROUND,W,read,write,now
from resource_lease import lease,wait_memory
PY='/mnt/ojc/miniconda3/envs/4danyone/bin/python'
SAM='/mnt/ojc/miniconda3/envs/4danyone-sam2/bin/python'
ROOT=ROUND/'experiments/moving_R01_foreground_v2'

async def main():
    with (ROUND/'moving_foreground.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=ROUND/'moving_foreground_queue.json'
        if path.exists():raise FileExistsError(path)
        state={'state':'waiting_for_gpu4_lease','pid':os.getpid(),'started_at':now(),'stages':[]}
        def save(**fields):state.update(fields,updated_at=now());write(path,state)
        save()
        try:
            async with lease([4]) as handles:
                await wait_memory([4],16000)
                if shutil.disk_usage(ROUND).free<30*1024**3:raise RuntimeError('Disk reserve below30GiB')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',FDANYONE_WAYMO_ROOT=str(ROOT),
                    FDANYONE_MOVING_ROOT=str(ROOT),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                    OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
                env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
                async def stage(name,python,script,*args):
                    cmd=[python,str(script),*args];log=ROUND/'logs'/f'moving_foreground_{name}.log'
                    with log.open('xb') as stream:
                        proc=await asyncio.create_subprocess_exec(*cmd,cwd=W.parent/'repo',env=env,stdout=stream,
                            stderr=asyncio.subprocess.STDOUT,pass_fds=(*handles,own.fileno()))
                        item={'stage':name,'pid':proc.pid,'command':cmd,'log':str(log),'state':'running'}
                        state['stages'].append(item);save(state='running',stage=name);code=await proc.wait()
                    item.update(exit_code=code,state='complete' if code==0 else 'failed');log.with_suffix('.exit').write_text(str(code)+'\n');save()
                    if code:raise RuntimeError(f'{name} failed; preserved {log}')
                await stage('prepare',PY,ROUND/'scripts/moving_foreground_prepare.py','prepare')
                await stage('segment',SAM,ROUND/'scripts/moving_segment.py')
                await stage('composite',PY,W/'scripts/prepare_foreground.py')
                await stage('finalize',PY,ROUND/'scripts/moving_foreground_prepare.py','finalize')
                await stage('mask_review',PY,W/'scripts/experiment_review.py')
                await stage('mask_gate',PY,W/'scripts/batch_gate.py','mask')
                await stage('motion_triple',PY,ROUND/'scripts/moving_motion_triple.py')
                save(state='foreground_motion_triple_complete_visual_review_pending')
        except Exception as error:save(state='failed',error=f'{type(error).__name__}: {error}');raise

if __name__=='__main__':asyncio.run(main())
