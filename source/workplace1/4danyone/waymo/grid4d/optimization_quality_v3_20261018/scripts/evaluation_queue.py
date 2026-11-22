"""Bounded asynchronous snapshot validation/evaluation, waiting on owned training only."""
import asyncio,json,os,sys,time,shutil
from pathlib import Path
NEW=Path(__file__).resolve().parents[1];ROOT=NEW.parent;status=dict(state='running',pid=os.getpid(),jobs={});queue=asyncio.Queue()
def save():
    status['updated_at_unix']=time.time();p=NEW/'EVALUATION_STATUS.json';q=p.with_suffix('.writing');q.write_text(json.dumps(status,indent=2)+'\n');q.replace(p)
def specifications():
    out=[]
    for label in ['N01','N07']:
        for stage,epochs in [('33k',[5000,15000,25000,33000]),('alpha',[38000,43000,48000]),('A',[49000,51000,53000]),('fused',[15000,20000,25000,33000])]:
            for epoch in epochs:out.append((label+'_'+stage,epoch,epoch==epochs[-1]))
    out.extend(('N04_Bmatch',x,x==53000) for x in [49000,51000,53000])
    for arm in ['C1','C2']:
        for epoch in [0,1,5,10,20]:out.append(('N11_'+arm,epoch,epoch==20))
    return out
async def producer():
    pending=specifications()
    while pending:
        for item in pending.copy():
            name,epoch,dense=item;run=NEW/'runs'/name;key=name+'_'+str(epoch)
            if (run/'snapshots'/f'{epoch}.json').exists():
                status['jobs'][key]=dict(state='queued',run=name,iteration=epoch,dense=dense);save();await queue.put(item);pending.remove(item)
            elif (run/'complete.json').exists():
                status['jobs'][key]=dict(state='not_produced',reason='Training terminal without this predeclared snapshot');save();pending.remove(item)
        if pending:await asyncio.sleep(20)
    await queue.put(None);await queue.put(None)
async def worker(gpu,uuid):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid)
    while True:
        item=await queue.get()
        if item is None:queue.task_done();return
        name,epoch,dense=item;run=NEW/'runs'/name;key=name+'_'+str(epoch);entry=status['jobs'][key];entry.update(state='running',gpu=gpu,uuid=uuid);save()
        try:
            assert shutil.disk_usage(NEW).free>35*2**30
            async def command(script,arguments,stage):
                with (NEW/'logs'/f'eval_{key}_{stage}.log').open('x') as log:
                    proc=await asyncio.create_subprocess_exec(sys.executable,str(script),*arguments,env=env,stdout=log,stderr=asyncio.subprocess.STDOUT);entry.update(stage=stage,pid=proc.pid);save();code=await proc.wait();assert code==0,(stage,code)
            await command(NEW/'scripts/validate_quality.py',['--run',str(run),'--iteration',str(epoch)],'validate')
            await command(NEW/'scripts/evaluate_quality.py',['--run',str(run),'--iteration',str(epoch)]+(['--dense'] if dense else []),'metrics')
            if dense:
                await command(NEW/'scripts/evaluate_parts.py',['--label',name[:3],'--arm',key],'parts')
                await command(ROOT/'scripts/replay_asset.py',['--run',str(run),'--iteration',str(epoch),'--output',str(NEW/'results'/key/'replay.mp4')],'video')
            entry['state']='complete';save()
        except Exception as error:entry.update(state='failed',error=repr(error));save()
        finally:queue.task_done()
async def main():
    await asyncio.gather(producer(),worker(5,'GPU-85bb9093-960d-9196-9cf8-cdecf8fd0765'),worker(4,'GPU-f9cfc0e8-a5ac-7168-2700-4b52fee1e21d'))
    status['state']='jobs_terminal';save()
asyncio.run(main())
