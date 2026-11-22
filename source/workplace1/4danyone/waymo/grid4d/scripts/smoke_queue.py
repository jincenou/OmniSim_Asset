"""Concurrent technical tests; each dependency chain is explicit and logged."""
import asyncio
import fcntl
import json
import os
import shutil
from baseline_queue import ROOT, gpu_info, save, now


async def main():
    state=dict(state='running',jobs={},pid=os.getpid(),started_at=now(),quality_experiment=False)
    path=ROOT/'research/smoke_queue.json'
    assert not path.exists()
    async def case(label,index,name,resume_test=False):
        with (ROOT/'cache/gpu_locks'/f'gpu{index}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            gpu=(await gpu_info())[index]
            assert gpu['free_mib']>18000 and shutil.disk_usage(ROOT).free>40*2**30
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'])
            config=ROOT/'configs'/f'{label}_smoke.json'
            run=ROOT/'runs'/name
            base=['/mnt/ojc/miniconda3/envs/grid4d/bin/python','-u',str(ROOT/'scripts/train_grid.py'),
                  '--config',str(config),'--run',str(run)]
            async def execute(stage,extra):
                key=name+'_'+stage;log=ROOT/'logs'/f'{key}.log'
                command=base+extra
                with log.open('x') as out:
                    child=await asyncio.create_subprocess_exec(*command,cwd=ROOT,env=env,stdout=out,stderr=asyncio.subprocess.STDOUT)
                    rec=dict(state='running',pid=child.pid,gpu=gpu,command=command,log=str(log),started_at=now())
                    state['jobs'][key]=rec;save(path,state)
                    code=await child.wait()
                rec.update(exit_code=code,state='complete' if code==0 else 'failed',ended_at=now());save(path,state)
                assert code==0,key
            if resume_test:
                await execute('prefix',['--stop-after','60'])
                await execute('resume',['--resume'])
            else:
                await execute('full',[])
    results=await asyncio.gather(case('N04',6,'N04_smoke_full_v1'),case('N04',7,'N04_smoke_resume_v1',True),
                                 case('N11',5,'N11_smoke_v1'),return_exceptions=True)
    state.update(state='training_tests_complete_replay_pending' if not any(isinstance(v,Exception) for v in results) else 'failed_requires_review',
                 errors=[repr(v) for v in results if isinstance(v,Exception)],ended_at=now())
    save(path,state)


if __name__=='__main__':
    asyncio.run(main())
