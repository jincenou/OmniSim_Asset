"""Wait for complete training/replay, evaluate asynchronously, then publish complete comparisons."""
import asyncio
import fcntl
import json
import os
from baseline_queue import ROOT, gpu_info, now, save

PYTHON='/mnt/ojc/miniconda3/envs/grid4d/bin/python'


async def main():
    path=ROOT/'optimization/EVALUATION_STATUS.json';assert not path.exists()
    state=dict(state='running',pid=os.getpid(),started_at=now(),jobs={});save(path,state)
    cpu=asyncio.Semaphore(2)
    async def command(record,args,name,env):
        log=ROOT/'logs'/f'optimization_eval_{name}.log'
        with log.open('x') as stream:
            child=await asyncio.create_subprocess_exec(PYTHON,'-u',str(ROOT/'scripts'/args[0]),*args[1:],cwd=ROOT,env=env,stdout=stream,stderr=asyncio.subprocess.STDOUT)
            record.update(pid=child.pid,log=str(log));save(path,state);code=await child.wait()
        record['exit_code']=code;save(path,state);assert code==0,str(log)
    async def job(label,arm,index):
        key=f'{label}_{arm}';record=dict(state='waiting_for_training_and_replay',gpu_index=index);state['jobs'][key]=record;save(path,state)
        try:
            while True:
                training=json.loads((ROOT/'optimization/STATUS.json').read_text());entry=training['jobs'].get(key,{})
                if entry.get('state')=='trained_replayed_quality_pending':break
                assert entry.get('state')!='failed',entry
                if training['state'] in ['partial_requires_review','training_replay_complete_quality_pending']:
                    raise RuntimeError(f'Terminal training queue without required successful replay: {entry}')
                await asyncio.sleep(15)
            with (ROOT/'cache/gpu_locks'/f'gpu{index}.lock').open('a') as lock:
                while True:
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:await asyncio.sleep(15);continue
                    gpu=(await gpu_info())[index]
                    if gpu['free_mib']>=6000:break
                    fcntl.flock(lock,fcntl.LOCK_UN);await asyncio.sleep(15)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid']);env.pop('PYTHONPATH',None)
                record.update(state='evaluating_fixed_final',gpu=gpu);save(path,state)
                await command(record,['evaluate_optimization.py','--label',label,'--arm',arm],key,env)
            async with cpu:
                record.update(state='evaluating_parts_and_time');save(path,state)
                await command(record,['part_metrics.py','--label',label,'--method',f'grid4d_{arm}'],key+'_parts',dict(os.environ,CUDA_VISIBLE_DEVICES=''))
            record.update(state='complete',ended_at=now());save(path,state)
        except Exception as error:
            record.update(state='failed',error=str(error),ended_at=now());save(path,state)
    await asyncio.gather(job('N04','continue',6),job('N11','continue',7),job('N04','parts',0),job('N11','parts',1),job('N04','alpha',2),job('N11','alpha',3),job('N04','combined',4),job('N11','combined',5))
    if all(r['state']=='complete' for r in state['jobs'].values()):
        state['state']='publishing_complete_comparisons';save(path,state)
        publisher={};state['publisher']=publisher
        await command(publisher,['publish_optimization.py'],'publish',dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        state.update(state='complete_quality_requires_visual_review',ended_at=now())
    else:state.update(state='partial_requires_review',ended_at=now())
    save(path,state)


if __name__=='__main__':asyncio.run(main())
