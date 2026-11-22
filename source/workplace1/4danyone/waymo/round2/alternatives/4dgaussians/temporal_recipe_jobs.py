"""Two native-setting smoke gates, then four async independent33k controls."""
import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

A = Path(__file__).resolve().parent
R = A.parents[1]
sys.path.insert(0, str(R / 'scripts'))
from resource_lease import lease, wait_memory

STATE = {'state': 'starting', 'pid': os.getpid(), 'jobs': {}}
STATE_PATH = A / 'temporal_recipe_queue.json'


def write():
    STATE['updated_at'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    path = STATE_PATH
    tmp = path.with_suffix('.writing')
    tmp.write_text(json.dumps(STATE, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


async def command(key, run, gpu, arguments, handles, environment):
    path = A / 'logs' / f'{key}.log'
    args = ['/mnt/ojc/miniconda3/envs/4dgaussians/bin/python', *map(str, arguments)]
    with path.open('xb') as log:
        process = await asyncio.create_subprocess_exec(*args, cwd=A, env=environment,
            stdout=log, stderr=asyncio.subprocess.STDOUT, pass_fds=handles)
        STATE['jobs'][key] = {'state': 'running', 'run': run, 'physical_gpu': gpu,
            'pid': process.pid, 'command': args, 'log': str(path),
            'started_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}
        write()
        code = await process.wait()
    (A / 'logs' / f'{key}.exit').write_text(str(code) + '\n')
    STATE['jobs'][key].update(state='complete' if code == 0 else 'failed', exit_code=code,
        ended_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
    write()
    if code:
        raise RuntimeError(f'{key} exit{code};inspect{path}')


async def recipe(label, run, control, gpu, smoke=False):
    assert (R / 'demo_jobs' / f'{label}.json').is_file()
    STATE['jobs'][run] = {'state': 'waiting_resources', 'physical_gpu': gpu}
    write()
    async with lease([gpu]) as handles:
        await wait_memory([gpu], 12000)
        assert shutil.disk_usage(A).free > 20 * 2**30
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
            CUDA_HOME='/mnt/ojc/miniconda3/envs/4dgaussians', PYTHONNOUSERSITE='1',
            OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
            TORCH_HOME=str(A / 'cache/torch'), MPLCONFIGDIR=str(A / 'cache/matplotlib'))
        args = [A / 'run_train.py', '--label', label, '--run', run, '--temporal-control', control]
        if smoke:
            args += ['--smoke']
        await command(run, run, gpu, args, handles, env)
        root = A / 'runs' / run
        await command(run + '_evaluation', run, gpu, [A / 'evaluate.py', '--run', root], handles, env)
        if not smoke:
            await command(run + '_profile', run, gpu, [A / 'quality_profile.py', '--run', root], handles, env)
            await command(run + '_demo', run, gpu, [A / 'render_demos.py', '--run', root,
                '--job', R / 'demo_jobs' / f'{label}.json'], handles, env)
            await command(run + '_comparison', run, gpu,
                [A / 'compare_temporal_controls.py', '--run', root], handles, env)


async def gate_results(values, next_state):
    failures = [str(item) for item in values if isinstance(item, Exception)]
    STATE.update(state='failed' if failures else next_state, failures=failures)
    write()
    if failures:
        raise RuntimeError('; '.join(failures))


async def main():
    assert not (A / 'temporal_recipe_queue.json').exists(), 'Do not restart completed or live queue'
    write()
    smoke = await asyncio.gather(
        recipe('m02_short61_24', 'm02_grid61_smoke_v1', 'grid61', 4, True),
        recipe('m02_short61_24', 'm02_smooth001_smoke_v1', 'smooth001', 6, True), return_exceptions=True)
    await gate_results(smoke, 'two_native_smokes_passed_four_full_recipes_running')
    full = await asyncio.gather(
        recipe('m02_short61_24', 'm02_61_grid61_recipe30k_v1', 'grid61', 4),
        recipe('r07_short61_24', 'r07_61_grid61_recipe30k_v1', 'grid61', 5),
        recipe('m02_short61_24', 'm02_61_smooth001_recipe30k_v1', 'smooth001', 6),
        recipe('r07_short61_24', 'r07_61_smooth001_recipe30k_v1', 'smooth001', 7), return_exceptions=True)
    await gate_results(full, 'four_native_temporal_controls_trained_replayed_profiled_rendered_and_compared_review_pending')


if __name__ == '__main__':
    asyncio.run(main())
