import asyncio,json,os,sys
from pathlib import Path
NEW=Path(__file__).resolve().parents[1];status={'state':'running','jobs':{}};locks=[asyncio.Lock(),asyncio.Lock()];devices=['GPU-0db9aa59-0293-a3ea-06a0-395913880fbd','GPU-4e635188-f5de-33f9-6ecb-53555a0a7c9d']
def save():
    p=NEW/'PROFILING_STATUS.json';q=p.with_suffix('.writing');q.write_text(json.dumps(status,indent=2)+'\n');q.replace(p)
async def job(name,iteration,gpu):
    async with locks[gpu]:
        run=NEW/'runs'/name;e=status['jobs'][name]={'state':'waiting','gpu':gpu};save()
        while not (run/'complete.json').exists():await asyncio.sleep(15)
        with (NEW/'logs'/f'{name}_profile.log').open('x') as log:
            proc=await asyncio.create_subprocess_exec(sys.executable,str(NEW/'scripts/profile_replay.py'),'--run',str(run),'--iteration',str(iteration),env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[gpu]),stdout=log,stderr=asyncio.subprocess.STDOUT);e.update(state='running',pid=proc.pid);save();e['exit_code']=await proc.wait()
        e['state']='complete' if e['exit_code']==0 else 'failed';save()
async def main():
    specs=[(label+'_'+stage,it) for label in ['N01','N07'] for stage,it in [('33k',33000),('alpha',48000),('A',53000),('fused',33000)]]+[('N04_Bmatch',53000),('N11_C1',20),('N11_C2',20)]
    await asyncio.gather(*(job(n,it,i%2) for i,(n,it) in enumerate(specs)));status['state']='jobs_terminal';save()
asyncio.run(main())
