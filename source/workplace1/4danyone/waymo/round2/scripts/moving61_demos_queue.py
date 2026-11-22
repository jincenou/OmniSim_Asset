"""Reviewed M0/M2 six-view diagnostics; wait for authentic foreground motion review."""
import asyncio
import fcntl
import os
import shutil
from pathlib import Path
from after_batch import ROUND, W, read, write, now
from resource_lease import lease, wait_memory

PY = '/mnt/ojc/miniconda3/envs/4danyone/bin/python'

async def main():
    with (ROUND/'moving61_demos.lock').open('a') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = ROUND/'moving61_demos_queue.json'
        if path.exists():
            raise FileExistsError(path)
        plan = read(ROUND/'moving61_demos_plan.json')
        state = {'state': 'running', 'pid': os.getpid(), 'started_at': now(), 'jobs': {}}
        def save():
            state['updated_at'] = now()
            write(path, state)
        save()

        async def case(job):
            name, gpu = job['id'], job['gpu']
            root = Path(job['output'])
            record = {'state': 'waiting_for_resources', 'gpu': gpu, 'output': str(root), 'stages': []}
            state['jobs'][name] = record
            save()
            try:
                source=Path(job['source'])
                while not (source/'motion_pair/review.json').exists():
                    upstream=read(ROUND/'moving61_queue_r2.json')
                    if upstream['state']=='failed':
                        raise RuntimeError('Foreground/motion upstream failed; inspect retained evidence')
                    record['state']='waiting_for_motion_visual_review';save()
                    await asyncio.sleep(30)
                review=read(source/'motion_pair/review.json')
                if not review['variants'][job['variant']]['allow_exploratory_generation']:
                    record['state']='motion_review_rejected';save();return
                record['state']='waiting_for_resources';save()
                if root.exists():
                    raise FileExistsError(root)
                async with lease([gpu]) as handles:
                    await wait_memory([gpu], plan['minimum_free_mib'])
                    if shutil.disk_usage(ROUND).free < plan['disk_reserve_gib']*1024**3:
                        raise RuntimeError('Insufficient reserved disk space')
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                               FDANYONE_WAYMO_ROOT=str(root), FDANYONE_SOURCE_EXPERIMENT=job['source'], FDANYONE_MOTION_VARIANT=job['variant'],
                               FDANYONE_ROOT_VIEWS='6', FDANYONE_RETAINED_VIEWS='6',
                               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
                    env.pop('PYTORCH_CUDA_ALLOC_CONF', None)

                    async def stage(label, script, *args):
                        command = [PY, str(script), *map(str, args)]
                        log = ROUND/'logs'/f'moving61_demo_{name}_{label}.log'
                        with log.open('xb') as stream:
                            proc = await asyncio.create_subprocess_exec(*command, cwd=W.parent/'repo', env=env,
                                stdout=stream, stderr=asyncio.subprocess.STDOUT,
                                pass_fds=(*handles, own.fileno()))
                            row = {'stage': label, 'pid': proc.pid, 'command': command, 'log': str(log),
                                   'state': 'running', 'started_at': now()}
                            record['stages'].append(row)
                            record.update(state='running', stage=label)
                            save()
                            code = await proc.wait()
                        log.with_suffix('.exit').write_text(str(code)+'\n')
                        row.update(state='complete' if code == 0 else 'failed', exit_code=code, ended_at=now())
                        save()
                        if code:
                            raise RuntimeError(f'{label} exited {code}; preserved {log}')

                    await stage('conditioning', ROUND/'scripts/moving61_conditioning.py')
                    await stage('generation', ROUND/'scripts/moving61_generate.py', 'smoke')
                    await stage('preview', W.parent/'scripts/visualize.py', root/'outputs/smoke')
                    await stage('validation', ROUND/'scripts/moving61_validate.py')
                    record.update(state='six_view_demo_complete_visual_review_pending',
                                  preview=str(root/'outputs/smoke/preview/multiview.mp4'))
            except Exception as error:
                record.update(state='failed', error=f'{type(error).__name__}: {error}')
            finally:
                save()

        await asyncio.gather(*map(case, plan['jobs']))
        failures = [name for name, record in state['jobs'].items() if record['state']=='failed']
        state.update(state='completed_with_failures' if failures else 'motion_pair_demos_finished_review_required',
                     failures=failures)
        save()
        if failures:
            raise SystemExit(1)

if __name__ == '__main__':
    asyncio.run(main())
