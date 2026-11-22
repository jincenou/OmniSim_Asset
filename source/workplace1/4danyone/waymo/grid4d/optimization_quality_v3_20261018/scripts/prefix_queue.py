import asyncio,fcntl,json,os,sys
from pathlib import Path
from datetime import datetime,timezone
NEW=Path(__file__).resolve().parents[1];ROOT=NEW.parent
status=dict(state='running',pid=os.getpid(),jobs={})
def save():
    p=NEW/'PREFIX_STATUS.json';q=p.with_suffix('.writing');q.write_text(json.dumps(status,indent=2)+'\n');q.replace(p)
async def job(label,gpu,uuid):
    entry=status['jobs'][label]=dict(state='starting',gpu_index=gpu,uuid=uuid,phases=[]);save()
    lock=(ROOT/'cache/gpu_locks'/f'gpu{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid)
    try:
        query=await asyncio.create_subprocess_exec('nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE)
        data,_=await query.communicate();assert query.returncode==0
        free={line.split(',')[0].strip():int(line.split(',')[1]) for line in data.decode().splitlines()};assert free[uuid]>16000
        run=NEW/'runs'/f'{label}_33k'
        phase=dict(name='native33k',started_at=datetime.now(timezone.utc).isoformat(),log=str(NEW/'logs'/f'{label}_prefix.log'))
        entry['phases'].append(phase)
        with Path(phase['log']).open('x') as log:
            process=await asyncio.create_subprocess_exec(sys.executable,str(NEW/'scripts/train_prefix.py'),'--config',str(NEW/'configs'/f'{label}_33k.json'),'--run',str(run),env=env,stdout=log,stderr=asyncio.subprocess.STDOUT)
            phase['pid']=process.pid;entry['state']='training';save();phase['exit_code']=await process.wait()
        phase['ended_at']=datetime.now(timezone.utc).isoformat();assert phase['exit_code']==0
        entry['state']='prefix_complete_continuation_pending';save()
    except Exception as error:
        entry.update(state='failed',error=repr(error));save()
    finally:lock.close()
async def main():
    await asyncio.gather(job('N01',6,'GPU-cd4b2342-fba4-2b42-beb9-60b698debc2d'),job('N07',7,'GPU-32040e1a-df6d-0776-5420-fae47bc11248'))
    status['state']='prefix_jobs_terminal';save()
asyncio.run(main())
