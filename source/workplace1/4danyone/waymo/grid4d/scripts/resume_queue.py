"""Same-prefix recovery test; never compare independently optimized prefixes."""
import asyncio
import fcntl
import os
from baseline_queue import ROOT, gpu_info,save,now


async def main():
    path=ROOT/'research/resume_queue.json';assert not path.exists()
    state=dict(state='running',jobs={},started_at=now(),pid=os.getpid())
    with (ROOT/'cache/gpu_locks/gpu6.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gpu=(await gpu_info())[6];assert gpu['free_mib']>18000
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'])
        python='/mnt/ojc/miniconda3/envs/grid4d/bin/python'
        config=str(ROOT/'configs/N04_smoke.json')
        a=ROOT/'runs/N04_smoke_full_v2';b=ROOT/'runs/N04_smoke_resume_v2'
        async def run(key,argv):
            log=ROOT/'logs'/f'{key}.log'
            with log.open('x') as output:
                proc=await asyncio.create_subprocess_exec(python,'-u',*argv,cwd=ROOT,env=env,stdout=output,stderr=asyncio.subprocess.STDOUT)
                rec=dict(state='running',pid=proc.pid,gpu=gpu,command=argv,log=str(log),started_at=now())
                state['jobs'][key]=rec;save(path,state);code=await proc.wait()
            rec.update(state='complete' if code==0 else 'failed',exit_code=code,ended_at=now());save(path,state)
            assert code==0,key
        await run('resume_full_v2',[str(ROOT/'scripts/train_grid.py'),'--config',config,'--run',str(a)])
        await run('resume_reload_v2',[str(ROOT/'scripts/train_grid.py'),'--config',config,'--run',str(b),'--resume-from',str(a/'recovery60.pt')])
        await run('resume_compare_v2',[str(ROOT/'scripts/check_resume.py')])
    state.update(state='complete',ended_at=now());save(path,state)


if __name__=='__main__':asyncio.run(main())
