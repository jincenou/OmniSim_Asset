import asyncio,json,os,sys
from pathlib import Path
NEW=Path(__file__).resolve().parents[1]
status=dict(state='running',pid=os.getpid(),jobs={})
def save():
    p=NEW/'CALIBRATION_STATUS.json';q=p.with_suffix('.writing');q.write_text(json.dumps(status,indent=2)+'\n');q.replace(p)
async def job(label,gpu,uuid):
    entry=status['jobs'][label]=dict(state='starting',gpu=gpu,uuid=uuid);save()
    try:
        with (NEW/'logs'/f'{label}_calibration.log').open('x') as log:
            proc=await asyncio.create_subprocess_exec(sys.executable,str(NEW/'scripts/calibrate_updates.py'),'--label',label,env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid),stdout=log,stderr=asyncio.subprocess.STDOUT)
            entry.update(state='calibrating',pid=proc.pid);save();entry['exit_code']=await proc.wait()
        entry['state']='complete' if entry['exit_code']==0 else 'failed';save()
    except Exception as e:entry.update(state='failed',error=repr(e));save()
async def main():
    await asyncio.gather(job('N04',0,'GPU-0db9aa59-0293-a3ea-06a0-395913880fbd'),job('N11',1,'GPU-4e635188-f5de-33f9-6ecb-53555a0a7c9d'))
    status['state']='jobs_terminal';save()
asyncio.run(main())
