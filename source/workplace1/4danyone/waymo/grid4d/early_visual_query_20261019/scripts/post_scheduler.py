"""Background export -> guarded fresh replay -> fixed metrics, bounded async GPU jobs."""
from scheduler import NEW,devices,save
import asyncio,os,json,time,shutil,fcntl
from pathlib import Path

def jobs_from_plan():
    p=json.loads((NEW/'PLAN.json').read_text());jobs=[]
    def add(key,script,args,done,deps,reserve=.05):jobs.append(dict(key=key,script=script,args=args,done=str(done),deps=[str(d) for d in deps],reserve=reserve))
    for l in p['labels']:
        key=l+'_warmup';add(key+'_metrics','evaluate',['--key',key,'--step','2999'],NEW/'results'/f'{key}_2999/metrics.json',[NEW/'runs'/key/'training.complete.json'])
        for arm in p['arms']:
            k=f'{l}_{arm}';run=NEW/'runs'/k;asset=NEW/'assets'/k
            add(k+'_export','export',['--key',k],asset/'export_validation.json',[run/'training.complete.json'],.5)
            add(k+'_replay','assets',['--asset',str(asset),'--expected',str(run/'replay_expected.npz')],asset/'replay_validation.json',[asset/'export_validation.json'])
            for step in [5000,15000,18000,32999]:
                result=NEW/'results'/f'{k}_{step}';deps=[run/f'snapshot_{step}.json']+([asset/'replay_validation.json'] if step==32999 else [])
                add(f'{k}_{step}_metrics','evaluate',['--key',k,'--step',str(step)],result/'metrics.json',deps,.3 if step==32999 else .05)
            finalkey=k+'_32999';cache=NEW/'cache/raw'/finalkey;out=NEW/'results'/finalkey
            add(k+'_parts','evaluate_parts',['--label',l,'--arm',finalkey],out/'parts.json',[cache/'complete.json'])
            if l in ['N06','N11']:add(k+'_expanded','evaluate_expanded',['--label',l,'--key',finalkey],out/'expanded.json',[cache/'complete.json'])
        if l in ['N06','N11']:
            old=p['inputs'][l]['old_references']
            for name,raw in [('A',old['A']['raw']),('fused',old['fused']['raw']),('HUST',str(NEW.parent/'tmp/baseline_raw'/l/'4dgs_grid61'))]:
                k=f'{l}_old_{name}';add(k+'_expanded','evaluate_expanded',['--label',l,'--key',k,'--raw',raw],NEW/'results'/k/'expanded.json',[Path(raw)/'complete.json'])
    return jobs

async def main():
    jobs=jobs_from_plan();running={};failed={};history=[];(NEW/'locks').mkdir(exist_ok=True)
    if (NEW/'post_history.json').exists():history=json.loads((NEW/'post_history.json').read_text())
    while True:
        for k,h in list(running.items()):
            if h['p'].returncode is not None:
                rc=h['p'].returncode;h['log'].close();h['lease'].close();history.append(dict(key=k,exit_code=rc,seconds=time.monotonic()-h['start'],gpu=h['gpu']['index']));save(NEW/'post_history.json',history)
                if rc:failed[k]=rc
                del running[k]
        free=shutil.disk_usage(NEW).free/2**30;minimum=json.loads((NEW/'PLAN.json').read_text())['resource']['minimum_disk_free_GiB'];train=json.loads((NEW/'STATUS.json').read_text());reserved=train.get('reserved_GiB',0)+sum(h['reserve'] for h in running.values());available=sorted(await devices(),key=lambda g:-g['free_GiB'])
        available=[g for g in available if g['free_GiB']>=12 and g['uuid'] not in {h['gpu']['uuid'] for h in running.values()}];waiting=[]
        for job in jobs:
            k=job['key']
            if Path(job['done']).exists() or k in running or k in failed:continue
            if not all(Path(d).exists() for d in job['deps']):continue
            if free-reserved-job['reserve']<minimum:waiting.append(dict(key=k,reason='disk_reservation'));continue
            chosen=None
            while available:
                gpu=available.pop(0);lease=(NEW/'locks'/f"gpu{gpu['index']}.lock").open('a')
                try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);chosen=(gpu,lease);break
                except BlockingIOError:lease.close()
            if chosen is None:waiting.append(dict(key=k,reason='GPU_resource'));continue
            gpu,lease=chosen;env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu['uuid'];attempt=len(list((NEW/'logs').glob(k+'_post*.log')));log=(NEW/'logs'/f'{k}_post{attempt}.log').open('x')
            proc=await asyncio.create_subprocess_exec('bash',str(NEW/'scripts/run.sh'),job['script'],*job['args'],env=env,stdout=log,stderr=asyncio.subprocess.STDOUT)
            running[k]=dict(p=proc,log=log,lease=lease,gpu=gpu,start=time.monotonic(),reserve=job['reserve']);reserved+=job['reserve'];print('started',k,gpu['index'],proc.pid,flush=True)
        done=[j['key'] for j in jobs if Path(j['done']).exists()];save(NEW/'POST_STATUS.json',dict(state='processing' if running else 'waiting_training_or_resources',completed=done,total_jobs=len(jobs),failed=failed,running={k:dict(pid=h['p'].pid,gpu=h['gpu']) for k,h in running.items()},waiting=waiting,controller_pid=os.getpid()))
        if len(done)==len(jobs):save(NEW/'post_queue.complete.json',dict(passed=True,jobs=done));return
        await asyncio.sleep(20)
if __name__=='__main__':
    with (NEW/'post_controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);asyncio.run(main())
