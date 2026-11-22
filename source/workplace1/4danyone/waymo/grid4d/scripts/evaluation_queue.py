"""Evaluate fixed snapshots independently; results never feed training choices."""
import asyncio
import fcntl
import os
import json
from baseline_queue import ROOT,gpu_info,save,now


async def main():
    path=ROOT/'EVALUATION_STATUS.json';assert not path.exists()
    state=dict(state='waiting_for_fixed_snapshots',started_at=now(),pid=os.getpid(),jobs={})
    save(path,state)
    async def sample(label,index):
        for iteration in [5000,15000,33000]:
            snapshot=ROOT/'runs'/f'{label}_33k_v1/snapshots/{iteration}.json'
            while not snapshot.exists():
                formal=json.loads((ROOT/'FORMAL_STATUS.json').read_text())
                if formal['runs'].get(label,{}).get('state')=='failed_requires_review':
                    raise RuntimeError(f'{label} training failed before snapshot{iteration}')
                await asyncio.sleep(30)
            with (ROOT/'cache/gpu_locks'/f'gpu{index}.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                gpu=(await gpu_info())[index]
                while gpu['free_mib']<6000:
                    await asyncio.sleep(30);gpu=(await gpu_info())[index]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'])
                key=f'{label}_{iteration}';log=ROOT/'logs'/f'evaluate_{key}.log'
                command=['/mnt/ojc/miniconda3/envs/grid4d/bin/python','-u',str(ROOT/'scripts/evaluate_grid.py'),'--label',label,'--iteration',str(iteration)]
                with log.open('x') as out:
                    proc=await asyncio.create_subprocess_exec(*command,cwd=ROOT,env=env,stdout=out,stderr=asyncio.subprocess.STDOUT)
                    record=dict(state='running',pid=proc.pid,gpu=gpu,command=command,log=str(log),started_at=now())
                    state['jobs'][key]=record;save(path,state);code=await proc.wait()
                record.update(state='complete' if code==0 else 'failed',exit_code=code,ended_at=now());save(path,state)
                assert code==0,key
    results=await asyncio.gather(sample('N04',0),sample('N11',1),return_exceptions=True)
    state.update(state='complete' if not any(isinstance(v,Exception) for v in results) else 'partial_requires_review',errors=[repr(v) for v in results if isinstance(v,Exception)],ended_at=now());save(path,state)


if __name__=='__main__':asyncio.run(main())
