"""Bounded asynchronous per-GPU orchestration with explicit receipts and failed-stage stops."""
import asyncio,json,os,shutil,sys,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]
ENV=Path('/mnt/ojc/miniconda3/envs/cem4dgs/bin/python')
OLD_ENV=Path('/mnt/ojc/miniconda3/envs/4dgaussians/bin/python')

def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

async def query_gpus():
    p=await asyncio.create_subprocess_exec('nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE)
    output=(await p.communicate())[0].decode();assert p.returncode==0
    def parse(line):
        i,u,m,b=map(str.strip,line.split(','));return {'physical_gpu':int(i),'uuid':u,'free_mib':int(m),'utilization':int(b)}
    return list(map(parse,output.strip().splitlines()))

async def execute(label,name,argv,gpu):
    assert shutil.disk_usage(ROOT).free>43*2**30,'Free space below admission margin'
    current=next(g for g in await query_gpus() if g['uuid']==gpu['uuid'])
    assert current['free_mib']>10000,'Insufficient memory for admission; no termination of other jobs'
    record={'label':label,'stage':name,'argv':list(map(str,argv)),'gpu':gpu,'started_at':stamp(),'state':'starting'}
    log=ROOT/'logs'/f'{label}_{name}.log';receipt=log.with_suffix('.json');assert not log.exists()
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu['uuid'];env['PYTHONUNBUFFERED']='1';env['PYTHONFAULTHANDLER']='1'
    with log.open('xb') as file:
        p=await asyncio.create_subprocess_exec(*map(str,argv),stdout=file,stderr=asyncio.subprocess.STDOUT,env=env,start_new_session=True)
        record.update(state='running',pid=p.pid,log=str(log));save(receipt,record)
        code=await p.wait();record.update(state='complete' if code==0 else 'failed',exit_code=code,ended_at=stamp());save(receipt,record)
    assert code==0,f'{label}/{name} failed exit {code}; see {log}'

async def case(label,gpu):
    await execute(label,'stage1',[ENV,'-u',ROOT/'scripts/train_case.py','--label',label,'--stage','1'],gpu)
    await execute(label,'stage2',[ENV,'-u',ROOT/'scripts/train_case.py','--label',label,'--stage','2'],gpu)
    await execute(label,'hust_raw',[OLD_ENV,'-u',ROOT/'scripts/baseline_cache.py','--label',label],gpu)
    await execute(label,'evaluate',[ENV,'-u',ROOT/'scripts/evaluate_case.py','--label',label],gpu)
    return {'label':label,'state':'evaluated_visual_review_pending','result':str(ROOT/'results'/label/'results.json')}

async def main():
    preflight=json.loads((ROOT/'research/preflight.json').read_text());assert preflight['state']=='passed'
    smoke=json.loads((ROOT/'smoke/N04/stage1/complete.json').read_text());assert smoke['kind']=='smoke_not_quality' and smoke['updates_this_stage']==1199
    gpus=[g for g in await query_gpus() if g['free_mib']>10000]
    gpus=sorted(gpus,key=lambda x:(x['utilization'],-x['free_mib'],x['physical_gpu']))[:3];assert len(gpus)==3
    labels=['N04','N08','N11'];state={'state':'three_case_pilot_running','started_at':stamp(),'pid':os.getpid(),
        'assignments':dict(zip(labels,gpus)),'scope':'CEM author two-stage reconstruction versus immutable HUST grid61; no frontend retraining'}
    save(ROOT/'STATUS.json',state)
    jobs=list(map(lambda pair:case(*pair),zip(labels,gpus)))
    results=await asyncio.gather(*jobs,return_exceptions=True)
    state.update(state='evaluated_visual_review_pending' if not any(isinstance(x,Exception) for x in results) else 'failed_partial',
        ended_at=stamp(),results=[str(x) if isinstance(x,Exception) else x for x in results]);save(ROOT/'STATUS.json',state)
    if state['state']=='failed_partial':raise RuntimeError(str(state['results']))

if __name__=='__main__':asyncio.run(main())
