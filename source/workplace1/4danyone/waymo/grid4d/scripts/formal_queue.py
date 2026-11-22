"""Two authorized clean33k runs followed by independent fixed-snapshot replay."""
import asyncio
import fcntl
import json
import os
import shutil
from baseline_queue import ROOT,gpu_info,save,now


async def main():
    state_path=ROOT/'FORMAL_STATUS.json';assert not state_path.exists()
    gate=json.loads((ROOT/'research/preflight.json').read_text());assert gate['formal_launch_allowed'] is True
    state=dict(state='running',started_at=now(),pid=os.getpid(),jobs={},runs={})
    async def case(label,index):
        run=ROOT/'runs'/f'{label}_33k_v1'
        with (ROOT/'cache/gpu_locks'/f'gpu{index}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            gpu=(await gpu_info())[index]
            assert gpu['free_mib']>=20000 and shutil.disk_usage(ROOT).free>40*2**30
            environment=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'])
            state['runs'][label]=dict(path=str(run),gpu=gpu,state='starting')
            async def execute(stage,arguments):
                log=ROOT/'logs'/f'{label}_33k_v1_{stage}.log'
                command=['/mnt/ojc/miniconda3/envs/grid4d/bin/python','-u',*arguments]
                with log.open('x') as out:
                    child=await asyncio.create_subprocess_exec(*command,cwd=ROOT,env=environment,stdout=out,stderr=asyncio.subprocess.STDOUT)
                    record=dict(state='running',pid=child.pid,gpu=gpu,command=command,log=str(log),started_at=now())
                    state['jobs'][label+'_'+stage]=record;state['runs'][label]['state']=stage
                    save(state_path,state);code=await child.wait()
                record.update(state='complete' if code==0 else 'failed',exit_code=code,ended_at=now());save(state_path,state)
                if code:
                    state['runs'][label]['state']='failed_requires_review';save(state_path,state)
                    raise RuntimeError(str(log))
            await execute('train',[str(ROOT/'scripts/train_grid.py'),'--config',str(ROOT/'configs'/f'{label}.json'),'--run',str(run)])
            for iteration in [5000,15000,33000]:
                await execute(f'replay_{iteration}',[str(ROOT/'scripts/validate_grid.py'),'--run',str(run),'--iteration',str(iteration)])
            state['runs'][label]['state']='trained_independently_replayed_quality_evaluation_pending';save(state_path,state)
    results=await asyncio.gather(case('N04',6),case('N11',7),return_exceptions=True)
    state.update(state='training_and_replay_complete_quality_pending' if not any(isinstance(v,Exception) for v in results) else 'partial_requires_review',
        errors=[repr(v) for v in results if isinstance(v,Exception)],ended_at=now());save(state_path,state)


if __name__=='__main__':asyncio.run(main())
