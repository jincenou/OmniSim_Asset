import asyncio,json,os,sys
from pathlib import Path
NEW=Path(__file__).resolve().parents[1];status=dict(state='running',pid=os.getpid(),jobs={})
def save():
    p=NEW/'COLOR_STATUS.json';q=p.with_suffix('.writing');q.write_text(json.dumps(status,indent=2)+'\n');q.replace(p)
async def job(arm,gpu,uuid):
    e=status['jobs'][arm]=dict(state='starting',gpu=gpu,uuid=uuid);save()
    try:
        assert json.loads((NEW/'color_preflight.json').read_text())['state']=='passed'
        with (NEW/'logs'/f'N11_{arm}.log').open('x') as log:
            proc=await asyncio.create_subprocess_exec(sys.executable,str(NEW/'scripts/train_color.py'),'--arm',arm,env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid),stdout=log,stderr=asyncio.subprocess.STDOUT)
            e.update(state='training',pid=proc.pid);save();e['exit_code']=await proc.wait()
        e['state']='training_complete_evaluation_pending' if e['exit_code']==0 else 'failed';save()
    except Exception as error:e.update(state='failed',error=repr(error));save()
async def main():
    await asyncio.gather(job('C1',0,'GPU-0db9aa59-0293-a3ea-06a0-395913880fbd'),job('C2',1,'GPU-4e635188-f5de-33f9-6ecb-53555a0a7c9d'))
    status['state']='jobs_terminal';save()
asyncio.run(main())
