"""Calibrated moving-camera diagnostic on a leased GPU, detached in tmux."""
import asyncio
import argparse
import fcntl
import os
import shutil
from after_batch import ROUND,read,write,now
from resource_lease import lease,wait_memory

PY='/mnt/ojc/miniconda3/envs/4danyone/bin/python'


async def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume-motion',action='store_true');parser.add_argument('--attempt',type=int,default=1);args=parser.parse_args()
    with (ROUND/'moving_queue.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        statepath=ROUND/(f'moving_queue_retry{args.attempt}.json' if args.resume_motion else 'moving_queue.json')
        if statepath.exists():raise FileExistsError(statepath)
        root=ROUND/'experiments/moving_R01_v1'
        if read(root/'native.json')['state']!='native_undistorted_moving_crop_ready':raise ValueError('Native preparation incomplete')
        state={'state':'waiting_for_gpu_lease','pid':os.getpid(),'started_at':now(),'gpu':4,'stages':[],
            'scope':'Moving-camera motion diagnostic; do not start generation before projection review',
            'shared_with_other_projects':True}
        def save(**fields):state.update(fields,updated_at=now());write(statepath,state)
        save()
        async with lease([4]) as fds:
            await wait_memory([4],20000)
            if shutil.disk_usage(ROUND).free<20*1024**3:raise RuntimeError('Disk reserve below20GiB')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
            env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
            async def stage(name,script):
                log=ROUND/'logs'/f'moving_R01_{name}{"_retry"+str(args.attempt) if args.resume_motion else ""}.log';command=[PY,str(ROUND/'scripts'/script)]
                with log.open('xb') as stream:
                    child=await asyncio.create_subprocess_exec(*command,env=env,stdout=stream,stderr=asyncio.subprocess.STDOUT,pass_fds=(*fds,own.fileno()))
                    record={'name':name,'pid':child.pid,'command':command,'log':str(log),'started_at':now(),'state':'running'}
                    state['stages'].append(record);save(state=name);code=await child.wait()
                log.with_suffix('.exit').write_text(str(code)+'\n');record.update(exit_code=code,state='complete' if code==0 else 'failed',ended_at=now());save()
                if code:save(state='failed');raise RuntimeError(f'{name} failed; preserve output and inspect log')
            if args.resume_motion:
                previous=read(ROUND/'moving_queue.json')
                if previous['state']!='failed' or previous['stages'][0]['exit_code']!=0:raise ValueError('No completed interpolation to resume')
                failed=root/f'motion_pair_failed_attempt{args.attempt}'
                if failed.exists():raise FileExistsError(failed)
                (root/'motion_pair').rename(failed)
                state['resumed_from']=str(ROUND/'moving_queue.json');save()
            else:
                await stage('interpolation','moving_interpolate.py')
            await stage('motion_pair','moving_motion.py')
            if read(root/'motion_pair/comparison.json')['state']!='motion_pair_inferred_projection_review_pending':raise ValueError('Missing motion diagnostic output')
            save(state='motion_pair_complete_visual_review_pending')


if __name__=='__main__':asyncio.run(main())
