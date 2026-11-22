"""Eight authorized continuation arms; independent final replay after training."""
import asyncio
import fcntl
import json
import os
import shutil
from baseline_queue import ROOT,gpu_info,save,now


async def main():
    path=ROOT/'optimization/STATUS.json';assert not path.exists()
    state=dict(state='preparing',jobs={},pid=os.getpid(),started_at=now());save(path,state)
    async def job(label,arm,index):
        key=f'{label}_{arm}';record=dict(state='waiting_for_train_only_core_cache',gpu_index=index)
        state['jobs'][key]=record;save(path,state)
        while not (ROOT/'optimization/data'/label/'complete.json').exists():
            preparation=json.loads((ROOT/'optimization/PREPARE_STATUS.json').read_text())
            assert preparation['state']!='failed',preparation
            await asyncio.sleep(10)
        if arm in ['alpha','combined']:
            record['state']='waiting_for_alpha_gradient_acceptance';save(path,state)
            while not (ROOT/'optimization/alpha_gradient_acceptance.json').exists():
                await asyncio.sleep(15)
            assert json.loads((ROOT/'optimization/alpha_gradient_acceptance.json').read_text())['state']=='passed'
        with (ROOT/'cache/gpu_locks'/f'gpu{index}.lock').open('a') as lock:
            while True:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    await asyncio.sleep(15);continue
                gpu=(await gpu_info())[index]
                if gpu['free_mib']>=16000 and shutil.disk_usage(ROOT).free>45*2**30:break
                fcntl.flock(lock,fcntl.LOCK_UN);record.update(state='waiting_for_memory',gpu=gpu);save(path,state);await asyncio.sleep(20)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid']);env.pop('PYTHONPATH',None)
            log=ROOT/'logs'/f'optimize_{key}.log'
            command=['/mnt/ojc/miniconda3/envs/grid4d/bin/python','-u',str(ROOT/'scripts/optimize_grid.py'),'--label',label,'--arm',arm]
            with log.open('x') as out:
                child=await asyncio.create_subprocess_exec(*command,cwd=ROOT,env=env,stdout=out,stderr=asyncio.subprocess.STDOUT)
                record.update(state='training',pid=child.pid,gpu=gpu,command=command,log=str(log),started_at=now());state['state']='running';save(path,state);code=await child.wait()
            record.update(state='trained' if code==0 else 'failed',exit_code=code,training_ended_at=now());save(path,state)
            if code:return
            run=ROOT/'optimization/runs'/key
            for iteration in [38000,43000,48000]:
                command=['/mnt/ojc/miniconda3/envs/grid4d/bin/python','-u',str(ROOT/'scripts/validate_grid.py'),'--run',str(run),'--iteration',str(iteration)]
                replay_log=ROOT/'logs'/f'optimize_{key}_replay{iteration}.log'
                with replay_log.open('x') as out:
                    child=await asyncio.create_subprocess_exec(*command,cwd=ROOT,env=env,stdout=out,stderr=asyncio.subprocess.STDOUT)
                    record.update(state='replaying',pid=child.pid,replay_iteration=iteration,replay_log=str(replay_log));save(path,state);code=await child.wait()
                assert code==0, str(replay_log)
            record.update(state='trained_replayed_quality_pending',ended_at=now());save(path,state)
    await asyncio.gather(job('N04','continue',6),job('N11','continue',7),job('N04','parts',0),job('N11','parts',1),job('N04','alpha',2),job('N11','alpha',3),job('N04','combined',4),job('N11','combined',5))
    state.update(state='training_replay_complete_quality_pending' if all(j['state']=='trained_replayed_quality_pending' for j in state['jobs'].values()) else 'partial_requires_review',ended_at=now());save(path,state)


if __name__=='__main__':asyncio.run(main())
