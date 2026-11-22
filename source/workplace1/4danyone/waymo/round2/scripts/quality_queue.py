"""Bounded async quality controls after baseline and the first walking diagnostic."""
import asyncio
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
from after_batch import ROUND,W,B,gate,read,write,now

PY='/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python'
STATE=ROUND/'quality_queue.json'


class Queue:
    def __init__(self):
        self.state={'pid':os.getpid(),'state':'waiting_for_previous_jobs','started_at':now(),'jobs':{},
                    'purpose':'Same generated video, common-six initialization, 6/12/21 training-view controls'}
        self.fds=[]

    def save(self,**values):
        self.state.update(values,updated_at=now());write(STATE,self.state)

    async def free(self,gpu):
        process=await asyncio.create_subprocess_exec('nvidia-smi',f'--id={gpu}',
            '--query-gpu=memory.free','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE)
        data,_=await process.communicate()
        if process.returncode:raise RuntimeError('Cannot query authorized GPU')
        return int(data.strip())

    async def run(self,label,views,gpu,steps=5000):
        name=f'{label}_v{views}_s{steps}'
        out=ROUND/'quality'/name;log=ROUND/'logs'/f'{name}.log'
        record={'state':'waiting_for_memory','gpu':gpu,'output':str(out),'started_at':now()}
        self.state['jobs'][name]=record;self.save()
        while await self.free(gpu)<8000:await asyncio.sleep(60)
        if shutil.disk_usage(ROUND).free<30*1024**3:raise RuntimeError('Disk reserve below 30 GiB')
        if out.exists() or log.exists():raise FileExistsError(name)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',
            CUDA_HOME='/mnt/ojc/miniconda3/envs/asset_harvester_2026',
            TORCH_EXTENSIONS_DIR=str(W/'reconstruction/deps/torch_extensions'),TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='4')
        env['PATH']=env['CUDA_HOME']+'/bin:'+env['PATH'];env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        command=[PY,str(ROUND/'scripts/train_control.py'),'--label',label,'--views',str(views),'--steps',str(steps),'--output',str(out)]
        with log.open('xb') as handle:
            process=await asyncio.create_subprocess_exec(*command,cwd=ROUND/'backend',env=env,
                stdout=handle,stderr=asyncio.subprocess.STDOUT,pass_fds=tuple(self.fds))
            record.update(state='training',pid=process.pid,command=command,log=str(log));self.save()
            code=await process.wait();log.with_suffix('.exit').write_text(str(code)+'\n')
        record.update(exit_code=code,ended_at=now())
        if code:
            record.update(state='failed');self.save();raise RuntimeError(f'{name} exited {code}')
        completed=read(out/'run_complete.json')
        if completed['steps']!=steps or completed['sampling']['optimized_steps']!=steps:raise RuntimeError('Missing actual sample-count evidence')
        if not all((out/path).is_file() for path in (f'ckpts/ckpt_{steps-1}.pt',f'ckpt_{steps-1}.ftgs.ply',f'videos/traj_4d_step{steps-1}.mp4')):
            raise RuntimeError('Missing trained/exported artifacts')
        record.update(state='trained_exported_metrics_pending');self.save()

    async def worker(self,gpu,tasks):
        while True:
            try:label,views=tasks.get_nowait()
            except asyncio.QueueEmpty:return
            try:await self.run(label,views,gpu)
            except Exception as error:
                name=f'{label}_v{views}_s5000';self.state['jobs'][name].update(state='failed',error=str(error));self.save()
            finally:tasks.task_done()

    async def main(self):
        if STATE.exists():raise FileExistsError('Preserve previous queue evidence before retry')
        self.save()
        with (ROUND/'controller.lock').open('a') as root_lock,(B/'controller.lock').open('a') as baseline_lock:
            while True:
                status=read(B/'status.json');walking=read(ROUND/'next_run.json')
                if gate(status,read(B/'controller.json'),read(B/'control.json')) and walking['state'] in ('failed','six_view_complete_visual_review_pending'):
                    try:fcntl.flock(root_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:pass
                    else:
                        try:fcntl.flock(baseline_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:fcntl.flock(root_lock,fcntl.LOCK_UN)
                        else:break
                self.save(baseline_completed=status['completed_distinct_technical_assets'],walking_state=walking['state'])
                await asyncio.sleep(60)
            self.fds=[root_lock.fileno(),baseline_lock.fileno()]
            self.save(state='running')
            free=await asyncio.gather(*map(self.free,(4,5,6,7)))
            gpu=4+max(range(4),key=lambda index:free[index])
            await self.run('phone_full',6,gpu,steps=2)
            self.save(gpu_probe='two_actual_updates_exported')
            tasks=asyncio.Queue()
            for case in [('phone_full',6),('phone_full',12),('phone_full',21),('pilot_043_twentyfour',6),('pilot_043_twentyfour',12),('pilot_043_twentyfour',21)]:tasks.put_nowait(case)
            await asyncio.gather(*map(lambda device:self.worker(device,tasks),(4,5,6,7)))
            comparisons={}
            for label in ('phone_full','pilot_043_twentyfour'):
                names=[f'{label}_v{views}_s5000' for views in (6,12,21)]
                if all(self.state['jobs'][name]['state']=='trained_exported_metrics_pending' for name in names):
                    states=[read(ROUND/'quality'/name/'initial_parameters.json') for name in names]
                    if not states[0]==states[1]==states[2]:raise RuntimeError(f'{label}: initial parameter mismatch invalidates strict comparison')
                    comparisons[label]='identical initial parameters and normalization verified; evaluation pending'
            self.save(state='training_done_evaluation_pending',comparisons=comparisons,
                failed=sum(job['state']=='failed' for job in self.state['jobs'].values()))


async def main():
    # A dedicated lock prevents two waiting orchestrators from being launched.
    with (ROUND/'quality_queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if STATE.exists():raise FileExistsError('Preserve previous queue evidence before retry')
        runner=Queue()
        try:await runner.main()
        except Exception as error:
            runner.save(state='failed',error=f'{type(error).__name__}: {error}');raise


if __name__=='__main__':asyncio.run(main())
