"""Two bounded async jobs; no unattended expansion beyond this pair."""
import asyncio
import fcntl
import os
import shutil
from after_batch import ROUND, W, read, write, now
from resource_lease import lease, wait_memory

PY = '/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python'

async def main():
    path = ROUND / 'm02_duration_queue.json'
    with path.with_suffix('.lock').open('a') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists():
            raise FileExistsError(path)
        state = {'state': 'starting', 'pid': os.getpid(), 'started_at': now(), 'jobs': {}}
        def persist():
            state['updated_at'] = now()
            write(path, state)
        persist()
        def environment(gpu):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                CUDA_HOME='/mnt/ojc/miniconda3/envs/asset_harvester_2026',
                TORCH_EXTENSIONS_DIR=str(W/'reconstruction/deps/torch_extensions'),
                TORCH_CUDA_ARCH_LIST='8.9', MAX_JOBS='4', OMP_NUM_THREADS='4',
                MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            env['PATH'] = env['CUDA_HOME'] + '/bin:' + env['PATH']
            env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
            return env
        async def stage(name, script, args, gpu, handles):
            if shutil.disk_usage(ROUND).free < 35 * 1024**3:
                raise RuntimeError('Disk reserve below 35 GiB')
            log = ROUND / 'logs' / f'm02_duration_{name}.log'
            command = [PY, str(ROUND/'scripts'/script), *map(str, args)]
            with log.open('xb') as stream:
                child = await asyncio.create_subprocess_exec(*command, cwd=ROUND/'backend',
                    env=environment(gpu), stdout=stream, stderr=asyncio.subprocess.STDOUT,
                    pass_fds=(*handles, own.fileno()))
                state['jobs'][name] = {'state': 'running', 'pid': child.pid, 'gpu': gpu,
                    'command': command, 'log': str(log), 'started_at': now()}
                persist()
                code = await child.wait()
            log.with_suffix('.exit').write_text(str(code)+'\n')
            state['jobs'][name].update(state='complete' if code == 0 else 'failed',
                                      exit_code=code, ended_at=now())
            persist()
            if code:
                raise RuntimeError(f'{name} exited {code}; {log}')
        async def case(job):
            sigma, gpu = job
            async with lease([gpu]) as handles:
                await wait_memory([gpu], 6000)
                await stage(f'sigma{sigma}', 'm02_duration_train.py', ['--sigma', sigma], gpu, handles)
                out = ROUND/'diagnostics/m02_duration'/f'sigma{sigma}'
                await stage(f'sigma{sigma}_metrics', 'evaluate_asset61.py',
                    ['--checkpoint', out/'ckpts/ckpt_14999.pt', '--data',
                     ROUND/'reconstruction/data/m02_short61_24', '--output', out/'quality_metrics'], gpu, handles)
        try:
            state['state'] = 'training_and_evaluation'; persist()
            results = await asyncio.gather(*map(case, [(15, 4), (8, 5)]), return_exceptions=True)
            errors = [str(item) for item in results if isinstance(item, BaseException)]
            if errors:
                raise RuntimeError('; '.join(errors))
            root = ROUND/'diagnostics/m02_duration'
            a, b = map(lambda sigma: read(root/f'sigma{sigma}/initial_parameters.json'), [15, 8])
            changed = [key for key in a if a[key] != b[key]]
            assert changed == ['durations'], changed
            a, b = map(lambda sigma: read(root/f'sigma{sigma}/experiment.json')['config'], [15, 8])
            diff = {key: [a[key], b[key]] for key in a if a[key] != b[key]}
            assert set(diff) == {'init_npz_path', 'result_dir'}, diff
            write(root/'paired_audit.json', {'state': 'only_initial_durations_changed',
                'initial_parameters_changed': changed, 'config_differences': diff})
            async with lease([4]) as handles:
                await wait_memory([4], 6000)
                await stage('comparison', 'm02_duration_review.py', [], 4, handles)
            state['state'] = 'trained_evaluated_rendered_visual_review_pending'; persist()
        except Exception as error:
            state.update(state='failed', error=f'{type(error).__name__}: {error}'); persist()
            raise

if __name__ == '__main__':
    asyncio.run(main())
