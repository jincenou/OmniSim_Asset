"""Escalate the reviewed walking diagnostic after the current quality queue exits."""
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
from after_batch import ROUND,W,B,P,PY,read,write,now,gate

ROOT=ROUND/'experiments/walk063_rootlocal24_v1'
STATE=ROUND/'walk24_run.json'


async def main():
    with (ROUND/'walk24.lock').open('a') as own,(ROUND/'controller.lock').open('a') as shared,(B/'controller.lock').open('a') as baseline:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if STATE.exists():raise FileExistsError(STATE)
        state={'state':'waiting_for_quality_queue','pid':os.getpid(),'started_at':now(),'output':str(ROOT),'stages':[]}
        def save(**fields):state.update(fields,updated_at=now());write(STATE,state)
        save()
        try:
            while True:
                review=read(ROUND/'reviews/walk063.json')
                if not review['allow_24_view_generation']:raise RuntimeError('Six-view visual gate does not allow escalation')
                quality=read(ROUND/'quality_queue.json')
                if quality['state'] in ('training_done_evaluation_pending','failed') and gate(read(B/'status.json'),read(B/'controller.json'),read(B/'control.json')):
                    try:fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:pass
                    else:
                        try:fcntl.flock(baseline,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:fcntl.flock(shared,fcntl.LOCK_UN)
                        else:break
                await asyncio.sleep(60)
            while True:
                process=await asyncio.create_subprocess_exec('nvidia-smi','--id=4,5,6,7','--query-gpu=memory.free','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE)
                data,_=await process.communicate()
                if process.returncode:raise RuntimeError('GPU query failed')
                if min(map(int,data.splitlines()))>=47000:break
                save(state='waiting_for_four_gpu_memory');await asyncio.sleep(60)
            if shutil.disk_usage(P).free<30*1024**3:raise RuntimeError('Disk reserve below30GiB')
            if ROOT.exists():raise FileExistsError(ROOT)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',
                FDANYONE_WAYMO_ROOT=str(ROOT),FDANYONE_ROOT_VIEWS='24',FDANYONE_RETAINED_VIEWS='24')
            env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
            async def stage(name,command):
                log=ROUND/'logs'/f'walk24_{name}.log'
                with log.open('xb') as handle:
                    process=await asyncio.create_subprocess_exec(*command,cwd=P/'repo',env=env,stdout=handle,
                        stderr=asyncio.subprocess.STDOUT,pass_fds=(own.fileno(),shared.fileno(),baseline.fileno()))
                    record={'stage':name,'pid':process.pid,'command':command,'gpu_ids':env['CUDA_VISIBLE_DEVICES'],'log':str(log),'started_at':now()}
                    state['stages'].append(record);save(state='running',stage=name)
                    code=await process.wait();log.with_suffix('.exit').write_text(str(code)+'\n')
                    record.update(exit_code=code,ended_at=now());save()
                    if code:raise RuntimeError(f'{name} exited {code}; see {log}')
            await stage('conditioning',[PY,str(ROUND/'scripts/walk_conditioning.py')])
            env['CUDA_VISIBLE_DEVICES']='4,5,6,7'
            await stage('generate',[PY,str(W/'scripts/generate.py'),'turbo'])
            out=ROOT/'outputs/turbo';shutil.copy2(ROOT/'preflight/root_motion.npz',out/'root_motion.npz')
            shutil.copy2(ROOT/'research/root_local.json',out/'root_local.json');camera=read(out/'cameras.json')
            if camera['world_frame']['name']!='root_local_human_world':raise RuntimeError('Wrong generated world frame')
            camera['root_motion']={'file':'root_motion.npz','sha256':hashlib.sha256((out/'root_motion.npz').read_bytes()).hexdigest(),
                'local_to_canonical_key':'local_to_canonical','time_key':'frame_times','motion_world_to_local_key':'motion_to_local','provenance':'root_local.json'}
            write(out/'cameras.json',camera)
            await stage('preview',[PY,str(P/'scripts/visualize.py'),str(out)])
            await stage('validate',[PY,str(W/'scripts/validate_output.py'),'turbo'])
            save(state='24_view_generated_visual_review_pending',preview=str(out/'preview/multiview.mp4'))
        except Exception as error:save(state='failed',error=f'{type(error).__name__}: {error}');raise


if __name__=='__main__':asyncio.run(main())
