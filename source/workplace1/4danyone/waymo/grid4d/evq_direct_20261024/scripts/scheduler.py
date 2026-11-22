"""Bounded asyncio subprocess queue; persistent tmux controller and UUID leases."""
import asyncio
import fcntl
import json
import os
import shutil
import time
from pathlib import Path

N=Path(__file__).resolve().parents[1]
STATES={}
HISTORY=[]
ACTIVE={}
LOCK=asyncio.Lock()


def save(path,value):
    tmp=path.with_suffix('.writing')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def status():
    save(N/'STATUS.json',dict(state='running' if ACTIVE else 'waiting_or_finished',at=time.time(),controller_pid=os.getpid(),jobs=STATES,total=12,complete=sum(x.get('state')=='complete' for x in STATES.values()),failed=sum(x.get('state')=='failed' for x in STATES.values()),disk_free_GiB=shutil.disk_usage(N).free/2**30,default_changed=False))


async def acquire(key):
    while True:
        async with LOCK:
            p=await asyncio.create_subprocess_exec('nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE)
            output,_=await p.communicate()
            if p.returncode:raise RuntimeError('GPU inspection failed')
            candidates=[dict(index=int(i),uuid=u.strip(),free_MiB=int(m),utilization=int(v)) for i,u,m,v in (r.split(',') for r in output.decode().splitlines())]
            candidates=sorted([g for g in candidates if g['uuid'] not in ACTIVE and g['free_MiB']>=16384],key=lambda g:(g['utilization'], -g['free_MiB'], g['index']))
            # Reserve enough for remaining checkpoints and complete-time evaluation.
            if shutil.disk_usage(N).free/2**30 > 5+7*(len(ACTIVE)+1):
                for gpu in candidates:
                    lease=(N/'locks'/f"gpu{gpu['index']}.lock").open('a')
                    try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:lease.close();continue
                    ACTIVE[gpu['uuid']]=key
                    return gpu,lease
            STATES[key]=dict(state='waiting_resources',minimum_GPU_free_GiB=16)
            status()
        await asyncio.sleep(20)


async def job(key):
    label,arm=key.split('_')
    gpu,lease=await acquire(key)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu['uuid']
    stages=[('train',['train','--label',label,'--arm',arm],N/'runs'/key/'training.complete.json'),
            ('export',['export','--key',key],N/'assets'/key/'export_validation.json'),
            ('replay',['assets','--asset',str(N/'assets'/key),'--expected',str(N/'runs'/key/'replay_expected.npz')],N/'assets'/key/'replay_validation.json'),
            ('evaluate',['evaluate','--key',key,'--step','32999'],N/'results'/f'{key}_32999/metrics.json'),
            ('parts',['evaluate_parts','--label',label,'--arm',f'{key}_32999'],N/'results'/f'{key}_32999/parts.json')]
    if label!='N04':stages.append(('expanded',['evaluate_expanded','--label',label,'--key',f'{key}_32999'],N/'results'/f'{key}_32999/expanded.json'))
    try:
        for stage,args,marker in stages:
            if marker.exists():continue
            if stage=='train' and (N/'runs'/key).exists():
                raise RuntimeError('Incomplete existing run: explicit recovery review required')
            if shutil.disk_usage(N).free<5*2**30:raise RuntimeError('Insufficient checkpoint write space')
            with (N/'logs'/f'{key}_{stage}.log').open('a') as log:
                began=time.time()
                process=await asyncio.create_subprocess_exec('bash',str(N/'scripts/run.sh'),*args,env=env,stdout=log,stderr=log)
                STATES[key]=dict(state='running',stage=stage,pid=process.pid,gpu=gpu,started=began)
                status();print('started',key,stage,gpu['index'],process.pid,flush=True)
                rc=await process.wait()
                HISTORY.append(dict(key=key,stage=stage,exit_code=rc,seconds=time.time()-began,gpu=gpu,marker=str(marker)))
                save(N/'queue_history.json',HISTORY)
                if rc or not marker.exists():raise RuntimeError(f'{stage} failed exit={rc}; see log')
        STATES[key].update(state='complete',ended=time.time())
    except Exception as error:
        STATES[key].update(state='failed',error=str(error));print('FAILED',key,error,flush=True)
    finally:
        del ACTIVE[gpu['uuid']];lease.close();status()


async def main():
    own=(N/'locks/controller.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((N/'PLAN.json').read_text());assert plan['state']=='frozen_before_training'
    assert json.loads((N/'research/preflight.json').read_text())['passed']
    assert json.loads((N/'research/portable_probe.json').read_text())['passed']
    if (N/'queue_history.json').exists():HISTORY.extend(json.loads((N/'queue_history.json').read_text()))
    # Round-robin identities ensures difficult motion enters the first GPU wave.
    keys=[f'{label}_{arm}' for arm in ('P3','P4','P2','P1') for label in ('N06','N11','N04')]
    await asyncio.gather(*(job(key) for key in keys))
    passed=all(x.get('state')=='complete' for x in STATES.values())
    save(N/'pipeline_queue.complete.json',dict(all_passed=passed,jobs=STATES))
    if passed:
        with (N/'logs/post.log').open('a') as log:
            process=await asyncio.create_subprocess_exec('bash',str(N/'scripts/run.sh'),'post',stdout=log,stderr=log)
            rc=await process.wait()
        save(N/'queue_exit.json',dict(training_evaluation_complete=True,post_exit_code=rc,visual_review='pending',default_changed=False))


if __name__=='__main__':asyncio.run(main())
