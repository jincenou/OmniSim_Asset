import asyncio,json,os,sys
from pathlib import Path
NEW=Path(__file__).resolve().parents[1];status=dict(state='running',pid=os.getpid(),jobs={})
def save():
    p=NEW/'FUSION_STATUS.json';q=p.with_suffix('.writing');q.write_text(json.dumps(status,indent=2)+'\n');q.replace(p)
async def job(label,gpu,uuid):
    e=status['jobs'][label]=dict(state='starting',gpu=gpu,uuid=uuid);save()
    try:
        with (NEW/'logs'/f'{label}_fused.log').open('x') as log:
            proc=await asyncio.create_subprocess_exec(sys.executable,str(NEW/'scripts/train_fused.py'),'--config',str(NEW/'configs'/f'{label}_fused.json'),'--run',str(NEW/'runs'/f'{label}_fused'),env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid),stdout=log,stderr=asyncio.subprocess.STDOUT)
            e.update(state='training',pid=proc.pid);save();e['exit_code']=await proc.wait()
        e['state']='training_complete_evaluation_pending' if e['exit_code']==0 else 'failed';save()
    except Exception as error:e.update(state='failed',error=repr(error));save()
async def main():
    await asyncio.gather(job('N01',2,'GPU-e3be373d-e235-e8d2-76bf-3559bea96b58'),job('N07',3,'GPU-16387ea4-8329-06c9-0c83-c18c0e71310c'))
    status['state']='jobs_terminal';save()
asyncio.run(main())
