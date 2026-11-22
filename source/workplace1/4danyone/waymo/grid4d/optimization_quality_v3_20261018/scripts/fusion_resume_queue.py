import asyncio,json,os,sys
from pathlib import Path
NEW=Path(__file__).resolve().parents[1];status=dict(state='running',pid=os.getpid(),jobs={})
def save():
    p=NEW/'FUSION_RESUME_STATUS.json';q=p.with_suffix('.writing');q.write_text(json.dumps(status,indent=2)+'\n');q.replace(p)
async def job(label,gpu,uuid):
    e=status['jobs'][label]=dict(state='starting',gpu=gpu,uuid=uuid);save()
    try:
        with (NEW/'logs'/f'{label}_fused_resume.log').open('x') as log:
            proc=await asyncio.create_subprocess_exec(sys.executable,str(NEW/'scripts/train_fused.py'),'--config',str(NEW/'configs'/f'{label}_fused.json'),'--run',str(NEW/'runs'/f'{label}_fused'),'--resume',env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid),stdout=log,stderr=asyncio.subprocess.STDOUT)
            e.update(state='training',pid=proc.pid);save();e['exit_code']=await proc.wait()
        e['state']='training_complete_evaluation_pending' if e['exit_code']==0 else 'failed';save()
    except Exception as error:e.update(state='failed',error=repr(error));save()
async def main():
    await asyncio.gather(job('N01',6,'GPU-cd4b2342-fba4-2b42-beb9-60b698debc2d'),job('N07',7,'GPU-32040e1a-df6d-0776-5420-fae47bc11248'))
    status['state']='jobs_terminal';save()
asyncio.run(main())
