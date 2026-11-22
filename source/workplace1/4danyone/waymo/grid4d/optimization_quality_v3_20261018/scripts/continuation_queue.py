"""Wait for owned prefixes, then execute exact alpha and A stages in tmux."""
import asyncio,fcntl,json,os,sys,shutil
from pathlib import Path
from datetime import datetime,timezone
NEW=Path(__file__).resolve().parents[1];ROOT=NEW.parent
status=dict(state='running',pid=os.getpid(),jobs={})
def save():
    status['updated_at']=datetime.now(timezone.utc).isoformat()
    p=NEW/'CONTINUATION_STATUS.json';q=p.with_suffix('.writing');q.write_text(json.dumps(status,indent=2)+'\n');q.replace(p)
async def job(label,gpu,uuid):
    entry=status['jobs'][label]=dict(state='waiting_for_prefix',gpu_index=gpu,uuid=uuid,phases=[]);save()
    lock=None
    try:
        while True:
            prefix=json.loads((NEW/'PREFIX_STATUS.json').read_text())['jobs'][label]
            if prefix['state']=='failed':raise RuntimeError('Prefix failed: '+str(prefix))
            if prefix['state']=='prefix_complete_continuation_pending':break
            await asyncio.sleep(20)
        lock=(ROOT/'cache/gpu_locks'/f'gpu{gpu}.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid)
        async def phase(stage):
            assert shutil.disk_usage(NEW).free>35*2**30
            record=dict(stage=stage,started_at=datetime.now(timezone.utc).isoformat(),log=str(NEW/'logs'/f'{label}_{stage}.log'))
            entry['phases'].append(record);entry['state']='training_'+stage;save()
            with Path(record['log']).open('x') as log:
                proc=await asyncio.create_subprocess_exec(sys.executable,str(NEW/'scripts/train_continuation.py'),'--label',label,'--stage',stage,env=env,stdout=log,stderr=asyncio.subprocess.STDOUT)
                record['pid']=proc.pid;save();record['exit_code']=await proc.wait()
            record['ended_at']=datetime.now(timezone.utc).isoformat();save()
            assert record['exit_code']==0,record
        await phase('alpha');await phase('A')
        entry['state']='A_complete_replay_evaluation_pending';save()
    except Exception as error:entry.update(state='failed',error=repr(error));save()
    finally:
        if lock:lock.close()
async def main():
    await asyncio.gather(job('N01',6,'GPU-cd4b2342-fba4-2b42-beb9-60b698debc2d'),job('N07',7,'GPU-32040e1a-df6d-0776-5420-fae47bc11248'))
    status['state']='jobs_terminal';save()
asyncio.run(main())
