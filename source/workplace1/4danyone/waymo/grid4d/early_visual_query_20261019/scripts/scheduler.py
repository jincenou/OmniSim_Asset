"""Bounded async subprocess queue; independent tmux, GPU UUID and disk reservation."""
import asyncio,json,os,time,shutil,fcntl
from pathlib import Path
from datetime import datetime,timezone
NEW=Path(__file__).resolve().parents[1]
def save(p,v):
    p=Path(p);tmp=p.with_suffix('.writing');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def size(p):return sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) if p.exists() else 0
async def devices():
    p=await asyncio.create_subprocess_exec('nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE)
    out,_=await p.communicate();assert p.returncode==0
    return [dict(index=int(a),uuid=b.strip(),free_GiB=float(c)/1024,utilization=int(d)) for a,b,c,d in [r.split(',') for r in out.decode().splitlines()]]
async def main():
    (NEW/'locks').mkdir(exist_ok=True)
    plan=json.loads((NEW/'PLAN.json').read_text());jobs=[dict(key=l+'_warmup',label=l,arm='R0',warmup=True,reserve=1.) for l in plan['labels']]+[dict(key=l+'_'+a,label=l,arm=a,warmup=False,reserve=6.) for a in plan['arms'] for l in plan['labels']]
    running={};failed={};history=[]
    if (NEW/'queue_history.json').exists():history=json.loads((NEW/'queue_history.json').read_text())
    while True:
        for k,h in list(running.items()):
            if h['process'].returncode is not None:
                rc=h['process'].returncode;h['log'].close();h['lease'].close();history.append(dict(key=k,device=h['device'],exit_code=rc,seconds=time.monotonic()-h['started'],at=datetime.now(timezone.utc).isoformat()));save(NEW/'queue_history.json',history)
                if rc!=0:failed[k]=rc
                del running[k]
        unfinished=[j for j in jobs if not (NEW/'runs'/j['key']/'training.complete.json').exists() and j['key'] not in failed]
        free=shutil.disk_usage(NEW).free/2**30;reserved=sum(max(0.,h['reserve']-size(NEW/'runs'/k)/2**30) for k,h in running.items());gpus=await devices();available=sorted([g for g in gpus if g['index'] in plan['resource']['authorized_physical_gpus'] and g['uuid'] not in {h['device']['uuid'] for h in running.values()} and g['free_GiB']>=12],key=lambda g:(-g['free_GiB'],g['index']))
        waiting=[]
        for j in unfinished:
            k=j['key']
            if k in running:continue
            if not j['warmup'] and not (NEW/'runs'/f"{j['label']}_warmup/training.complete.json").exists():waiting.append(dict(key=k,reason='cold_warmup_dependency'));continue
            if not available:waiting.append(dict(key=k,reason='GPU_free_memory_or_own_worker'));continue
            need=max(0.,j['reserve']-size(NEW/'runs'/k)/2**30)
            if free-reserved-need<plan['resource']['minimum_disk_free_GiB']:waiting.append(dict(key=k,reason='disk_reservation',free_GiB=free,other_reserved_GiB=reserved,need_GiB=need));continue
            run=NEW/'runs'/k;argv=['bash',str(NEW/'scripts/run.sh'),'train','--label',j['label'],'--arm',j['arm']]
            if j['warmup']:argv.append('--warmup')
            if run.exists():
                waiting.append(dict(key=k,reason='existing_incomplete_run_requires_explicit_resume_review'));continue
            gpu=available.pop(0);lease=(NEW/'locks'/f"gpu{gpu['index']}.lock").open('a')
            try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:lease.close();waiting.append(dict(key=k,reason='own_postprocessing_GPU_lease'));continue
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu['uuid'];attempt=len(list((NEW/'logs').glob(f'{k}*.log')));log=(NEW/'logs'/f'{k}_attempt{attempt}.log').open('x');p=await asyncio.create_subprocess_exec(*argv,env=env,stdout=log,stderr=asyncio.subprocess.STDOUT)
            running[k]=dict(process=p,log=log,lease=lease,device=gpu,started=time.monotonic(),reserve=j['reserve']);reserved+=need;print('started',k,gpu['index'],p.pid,flush=True)
        complete=[j['key'] for j in jobs if (NEW/'runs'/j['key']/'training.complete.json').exists()]
        save(NEW/'STATUS.json',dict(state='training' if running else 'waiting_resources_or_review',at=datetime.now(timezone.utc).isoformat(),controller_pid=os.getpid(),completed_jobs=complete,formal_terminals=sum(not k.endswith('warmup') for k in complete),required_terminals=12,running={k:dict(pid=h['process'].pid,device=h['device'],seconds=time.monotonic()-h['started']) for k,h in running.items()},failed=failed,waiting=waiting,disk_free_GiB=free,reserved_GiB=reserved,postprocessing='not yet complete; separately verified queue required'))
        if len(complete)+len(failed)==len(jobs) and not running:
            save(NEW/'training_queue.complete.json',dict(all_passed=not failed,completed_jobs=complete,failed=failed));return
        await asyncio.sleep(20)
if __name__=='__main__':asyncio.run(main())
