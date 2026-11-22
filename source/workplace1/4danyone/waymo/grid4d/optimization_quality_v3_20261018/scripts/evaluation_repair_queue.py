"""One retry only for diagnosed sample-order assertion; original logs retained."""
import asyncio,json,os,sys,time
from pathlib import Path
NEW=Path(__file__).resolve().parents[1];ROOT=NEW.parent
status=json.loads((NEW/'EVALUATION_REPAIR_STATUS.json').read_text()) if (NEW/'EVALUATION_REPAIR_STATUS.json').exists() else dict(state='running',jobs={})
status['state']='running';semaphore=asyncio.Semaphore(2);tasks=[];seen=set()
def save():
    p=NEW/'EVALUATION_REPAIR_STATUS.json';q=p.with_suffix('.writing');q.write_text(json.dumps(status,indent=2)+'\n');q.replace(p)
async def repair(key,item,gpu,uuid):
    async with semaphore:
        entry=status['jobs'][key];entry.update(state='running',gpu=gpu);save();run=NEW/'runs'/item['run'];epoch=item['iteration'];env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid)
        try:
            log=(NEW/'logs'/f'eval_{key}_metrics.log').read_text()
            known_linecache_change=key in ['N04_Bmatch_49000','N07_33k_33000'] and 'baseline=json.loads' in log and 'line 33' in log
            assert ('baseline[\'samples\']' in log or known_linecache_change) and 'AssertionError' in log,'Not the diagnosed ordering failure'
            async def command(script,args,stage):
                with (NEW/'logs'/f'repair_{key}_{stage}.log').open('x') as output:
                    proc=await asyncio.create_subprocess_exec(sys.executable,str(script),*args,env=env,stdout=output,stderr=asyncio.subprocess.STDOUT);entry.update(stage=stage,pid=proc.pid);save();code=await proc.wait();assert code==0,(stage,code)
            await command(NEW/'scripts/evaluate_quality.py',['--run',str(run),'--iteration',str(epoch)]+(['--dense'] if item['dense'] else []),'metrics')
            if item['dense']:
                await command(NEW/'scripts/evaluate_parts.py',['--label',run.name[:3],'--arm',key],'parts')
                await command(ROOT/'scripts/replay_asset.py',['--run',str(run),'--iteration',str(epoch),'--output',str(NEW/'results'/key/'replay.mp4')],'video')
            entry['state']='complete';save()
        except Exception as e:entry.update(state='failed',error=repr(e));save()
async def main():
    while True:
        original=json.loads((NEW/'EVALUATION_STATUS.json').read_text())
        for key,item in original['jobs'].items():
            if item['state']=='failed' and key not in seen and status['jobs'].get(key,{}).get('state')!='complete':
                seen.add(key);gpu=len(tasks)%2;uuid=['GPU-0db9aa59-0293-a3ea-06a0-395913880fbd','GPU-4e635188-f5de-33f9-6ecb-53555a0a7c9d'][gpu];previous=status['jobs'].get(key);status['jobs'][key]=dict(state='queued',previous_attempt=previous);save();tasks.append(asyncio.create_task(repair(key,item,gpu,uuid)))
        if original['state']=='jobs_terminal':break
        await asyncio.sleep(20)
    await asyncio.gather(*tasks);status['state']='jobs_terminal';save()
asyncio.run(main())
