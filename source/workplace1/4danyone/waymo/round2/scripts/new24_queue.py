"""Reviewed natural walking demos; shared four-device generation lease."""
import asyncio
import fcntl
import os
import shutil
from pathlib import Path
from after_batch import ROUND, W, read, write, now, sha
from resource_lease import lease, wait_memory

PY='/mnt/ojc/miniconda3/envs/4danyone/bin/python'

async def main():
    with (ROUND/'new24.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_path=ROUND/'new24_queue.json'
        if state_path.exists():raise FileExistsError(state_path)
        state={'state':'running','pid':os.getpid(),'started_at':now(),'jobs':{}}
        def save():
            state['updated_at']=now();write(state_path,state)
        save()
        async def case(job):
            root=Path(job['output']);name=job['id'];gpu=job['gpu']
            row={'state':'waiting','gpu':gpu,'output':str(root),'stages':[]};state['jobs'][name]=row;save()
            try:
                review=read(ROUND/'reviews/new_four_six.json')[job['six_view_review']]
                if not review['allow_24_view_generation']:raise ValueError('Review did not allow this run')
                if shutil.disk_usage(ROUND).free<40*1024**3:raise RuntimeError('Disk reserve below40GiB')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    FDANYONE_WAYMO_ROOT=str(root),FDANYONE_SOURCE_EXPERIMENT=job['source'],
                    FDANYONE_ROOT_VIEWS='24',FDANYONE_RETAINED_VIEWS='24',
                    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
                env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
                async def stage(label,script,*args,handles):
                    cmd=[PY,str(script),*map(str,args)];log=ROUND/'logs'/f'new24_{name}_{label}.log'
                    with log.open('xb') as stream:
                        proc=await asyncio.create_subprocess_exec(*cmd,cwd=W.parent/'repo',env=env,
                            stdout=stream,stderr=asyncio.subprocess.STDOUT,pass_fds=(*handles,own.fileno()))
                        item={'stage':label,'command':cmd,'pid':proc.pid,'log':str(log),'state':'running','started_at':now()}
                        row['stages'].append(item);row.update(state='running',stage=label);save();code=await proc.wait()
                    log.with_suffix('.exit').write_text(str(code)+'\n');item.update(exit_code=code,state='complete' if code==0 else 'failed',ended_at=now());save()
                    if code:raise RuntimeError(f'{label} exit{code}; see {log}')
                async with lease([gpu]) as handles:
                    await wait_memory([gpu],16000)
                    await stage('conditioning',ROUND/'scripts/new24_conditioning.py',handles=handles)
                row.update(state='waiting_for_four_gpu_memory',stage='generation');save()
                async with lease([4,5,6,7]) as handles:
                    await wait_memory([4,5,6,7],47000)
                    env['CUDA_VISIBLE_DEVICES']='4,5,6,7'
                    await stage('generation',ROUND/'scripts/new24_generate.py','turbo',handles=handles)
                out=root/'outputs/turbo'
                shutil.copy2(root/'preflight/root_motion.npz',out/'root_motion.npz')
                shutil.copy2(root/'research/root_local.json',out/'root_local.json')
                camera=read(out/'cameras.json')
                if camera['world_frame']['name']!='root_local_human_world':raise ValueError('Wrong camera frame')
                camera['root_motion']={'file':'root_motion.npz','sha256':sha(out/'root_motion.npz'),
                    'local_to_canonical_key':'local_to_canonical','time_key':'frame_times','motion_world_to_local_key':'motion_to_local','provenance':'root_local.json'}
                write(out/'cameras.json',camera)
                await stage('preview',W.parent/'scripts/visualize.py',out,handles=())
                await stage('validation',W/'scripts/validate_output.py','turbo',handles=())
                row.update(state='24view_complete_visual_review_pending',preview=str(out/'preview/multiview.mp4'))
            except Exception as error:row.update(state='failed',error=f'{type(error).__name__}: {error}')
            finally:save()
        await asyncio.gather(*map(case,read(ROUND/'next24_plan.json')['jobs']))
        state['failures']=[key for key,value in state['jobs'].items() if value['state']=='failed']
        state['state']='completed_with_failures' if state['failures'] else 'three_24view_demos_complete_review_pending';save()
        if state['failures']:raise SystemExit(1)

if __name__=='__main__':asyncio.run(main())
