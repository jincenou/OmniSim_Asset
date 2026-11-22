"""Wait for this exact validation process, then run smoke and the bounded pilot queue."""
import asyncio,json,os
from pathlib import Path
from run_queue import ROOT,ENV,execute,query_gpus,save,stamp,main as queue

def process_identity(pid):
    path=Path('/proc')/str(pid)/'stat'
    if not path.exists():return None
    tail=path.read_text().split(') ',1)[1].split()
    return None if tail[0]=='Z' else tail[19]

async def main():
    gate=json.loads((ROOT/'research/preflight_process.json').read_text());pid=gate['pid']
    save(ROOT/'STATUS.json',{'state':'waiting_for_exact_preflight_process','pid':os.getpid(),'preflight_pid':pid,'updated_at':stamp()})
    while process_identity(pid)==gate['start_ticks']:
        await asyncio.sleep(20)
    result=json.loads((ROOT/'research/preflight.json').read_text());assert result['state']=='passed'
    assert not (ROOT/'smoke/N04/stage1').exists()
    gpu=next(g for g in await query_gpus() if g['uuid']==gate['gpu_uuid'])
    save(ROOT/'STATUS.json',{'state':'smoke_training','pid':os.getpid(),'gpu':gpu,'updated_at':stamp()})
    await execute('N04','smoke',[ENV,'-u',ROOT/'scripts/train_case.py','--label','N04','--stage','1','--smoke'],gpu)
    await queue()

if __name__=='__main__':
    try:asyncio.run(main())
    except Exception as e:
        save(ROOT/'STATUS.json',{'state':'failed_requires_review','error':repr(e),'pid':os.getpid(),'updated_at':stamp()});raise
