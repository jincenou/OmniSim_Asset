"""Eight bounded asyncio GPU workers; complete training/evaluation before publication."""
import asyncio,fcntl,json,os,shutil,sys,time
from pathlib import Path
from datetime import datetime,timezone
NEW=Path(__file__).resolve().parents[1];ROOT=NEW.parent
PLAN=json.loads((NEW/'PLAN.json').read_text());status=dict(state='running',started_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),jobs={},gpu_policy=PLAN['runtime']);queue=asyncio.Queue()
UUIDS=['GPU-0db9aa59-0293-a3ea-06a0-395913880fbd','GPU-4e635188-f5de-33f9-6ecb-53555a0a7c9d','GPU-e3be373d-e235-e8d2-76bf-3559bea96b58','GPU-16387ea4-8329-06c9-0c83-c18c0e71310c','GPU-f9cfc0e8-a5ac-7168-2700-4b52fee1e21d','GPU-85bb9093-960d-9196-9cf8-cdecf8fd0765','GPU-cd4b2342-fba4-2b42-beb9-60b698debc2d','GPU-32040e1a-df6d-0776-5420-fae47bc11248']
publication=asyncio.Semaphore(2)
def save():
    status['updated_at']=datetime.now(timezone.utc).isoformat();p=NEW/'STATUS.json';q=p.with_suffix('.writing');q.write_text(json.dumps(status,indent=2,allow_nan=False)+'\n');q.replace(p)
async def command(entry,gpu,name,script,args):
    if shutil.disk_usage(NEW).free<40*2**30: raise RuntimeError('Disk below35GiB reserve plus5GiB write headroom; no historical deletion')
    path=NEW/'logs'/f'{name}.log'; assert not path.exists(),path
    stage=dict(name=name,state='starting',log=str(path),started_at=time.time());entry['stages'].append(stage);save()
    with path.open('x') as log:
        p=await asyncio.create_subprocess_exec(sys.executable,str(script),*args,env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUIDS[gpu]),stdout=log,stderr=asyncio.subprocess.STDOUT)
        stage.update(state='running',pid=p.pid);entry['current_stage']=name;save();code=await p.wait()
    stage.update(state='complete' if code==0 else 'failed',exit_code=code,ended_at=time.time());save()
    if code:raise RuntimeError(f'{name} exited {code}; see {path}')
async def evaluate(entry,gpu,label,stage,iteration,dense):
    run=NEW/'runs'/f'{label}_{stage}';key=run.name+'_'+str(iteration)
    await command(entry,gpu,key+'_validate',NEW/'scripts/validate_quality.py',['--run',str(run),'--iteration',str(iteration)])
    await command(entry,gpu,key+'_metrics',NEW/'scripts/evaluate_quality.py',['--run',str(run),'--iteration',str(iteration)]+(['--dense'] if dense else []))
    if dense:await command(entry,gpu,key+'_parts',NEW/'scripts/evaluate_parts.py',['--label',label,'--arm',key])
async def wait_resource(entry,gpu):
    while True:
        p=await asyncio.create_subprocess_exec('nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE)
        out,_=await p.communicate();assert p.returncode==0
        available={r.split(',')[0].strip():int(r.split(',')[1]) for r in out.decode().splitlines()}
        entry['free_memory_MiB_at_check']=available[UUIDS[gpu]];entry['free_disk_GiB_at_check']=shutil.disk_usage(NEW).free/2**30
        if available[UUIDS[gpu]]>=16000 and entry['free_disk_GiB_at_check']>=40:save();return
        entry['state']='waiting_resource';save();await asyncio.sleep(30)
async def worker(gpu):
    while True:
        item=await queue.get()
        if item is None:queue.task_done();return
        label,arm=item;key=label+'_'+arm;entry=status['jobs'][key];entry.update(state='assigned',gpu=gpu,uuid=UUIDS[gpu],stages=[]);save();lock=None
        try:
            await wait_resource(entry,gpu)
            lock=(ROOT/'cache/gpu_locks'/f'gpu{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            entry['state']='running';save()
            if arm=='A':
                await command(entry,gpu,label+'_native_train',NEW/'scripts/train_prefix.py',['--config',str(NEW/'configs'/f'{label}_33k.json'),'--run',str(NEW/'runs'/f'{label}_33k')])
                await command(entry,gpu,label+'_alpha_train',NEW/'scripts/train_continuation.py',['--label',label,'--stage','alpha'])
                await command(entry,gpu,label+'_A_train',NEW/'scripts/train_continuation.py',['--label',label,'--stage','A'])
                await evaluate(entry,gpu,label,'33k',33000,False)
                await evaluate(entry,gpu,label,'alpha',48000,False)
                await evaluate(entry,gpu,label,'A',53000,True)
            else:
                await command(entry,gpu,label+'_fused_train',NEW/'scripts/train_fused.py',['--config',str(NEW/'configs'/f'{label}_fused.json'),'--run',str(NEW/'runs'/f'{label}_fused')])
                await evaluate(entry,gpu,label,'fused',33000,True)
            entry['state']='complete';entry['ended_at']=time.time();save()
        except Exception as error:entry.update(state='failed',error=repr(error));save()
        finally:
            if lock:lock.close()
            queue.task_done()
async def publish_all():
    import publish_cross8
    pending=list(PLAN['labels'])
    while pending:
        ready=[label for label in pending if publish_cross8.ready(label)]
        async def one(label):
            async with publication:
                try:await asyncio.to_thread(publish_cross8.publish_label,label)
                except Exception as error:
                    status.setdefault('publication_errors',{})[label]=repr(error);save()
        if ready:
            await asyncio.gather(*(one(label) for label in ready));pending=[v for v in pending if v not in ready]
        await asyncio.to_thread(publish_cross8.overview)
        if not pending:return
        if all(j['state'] in ['complete','failed'] for j in status['jobs'].values()):
            status['unpublished_labels']=pending;save();return
        await asyncio.sleep(30)
async def main():
    # First four fullA and four fused jobs occupy all eight authorized devices.
    jobs=[('N02','A'),('N03','A'),('N06','A'),('N08','A'),('N02','fused'),('N03','fused'),('N06','fused'),('N08','fused'),('N04','fused'),('N11','fused')]
    for label,arm in jobs:status['jobs'][label+'_'+arm]=dict(state='queued');queue.put_nowait((label,arm))
    for _ in UUIDS:queue.put_nowait(None)
    save()
    await asyncio.gather(*(worker(gpu) for gpu in [6,7,0,1,2,3,4,5]),publish_all())
    status['state']='training_evaluation_complete_review_pending' if all(v['state']=='complete' for v in status['jobs'].values()) and not status.get('publication_errors') and not status.get('unpublished_labels') else 'terminal_with_recorded_failures';save()
if __name__=='__main__':
    with (NEW/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert shutil.disk_usage(NEW).free>=90*2**30,'Admission requires90GiB before all ten branches'
        asyncio.run(main())
