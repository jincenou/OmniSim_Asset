"""Detached 61-frame motion diagnostic with shared GPU lease and exit records."""
import asyncio
import fcntl
import os
from after_batch import ROUND,W,read,write,now
from resource_lease import lease,wait_memory
PY='/mnt/ojc/miniconda3/envs/4danyone/bin/python'
ROOT=ROUND/'experiments/moving_R01_short61_v1'

async def main():
    path=ROUND/'moving61_queue.json'
    with (ROUND/'moving61.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if path.exists():raise FileExistsError(path)
        state={'state':'waiting_for_gpu4','pid':os.getpid(),'started_at':now(),'stages':[]}
        def save(**kw):state.update(kw,updated_at=now());write(path,state)
        save()
        try:
            assert read(ROOT/'research/input_review.json')['allow_motion_diagnostic']
            assert read(ROOT/'research/batch_gate_mask.json')['passed']
            async with lease([4]) as handles:
                await wait_memory([4],16000)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    FDANYONE_MOVING_ROOT=str(ROOT),FDANYONE_WAYMO_ROOT=str(ROOT),
                    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',
                    OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
                env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
                cmd=[PY,str(ROUND/'scripts/moving61_motion.py')]
                log=ROUND/'logs/moving61_motion.log'
                with log.open('xb') as stream:
                    p=await asyncio.create_subprocess_exec(*cmd,cwd=W.parent/'repo',env=env,
                        stdout=stream,stderr=asyncio.subprocess.STDOUT,pass_fds=(*handles,own.fileno()))
                    item={'stage':'motion_triple','pid':p.pid,'command':cmd,'log':str(log),'started_at':now()}
                    state['stages'].append(item);save(state='running')
                    code=await p.wait()
                item.update(exit_code=code,ended_at=now());log.with_suffix('.exit').write_text(str(code)+'\n')
                if code:raise RuntimeError(f'Motion exited {code}; see {log}')
                save(state='motion_triple_complete_visual_review_pending')
        except Exception as e:save(state='failed',error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':asyncio.run(main())
