"""Four bounded GPU workers; retained models only, no training."""
import asyncio
from argparse import ArgumentParser
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

A = Path(__file__).resolve().parent
R = A.parents[1]
sys.path.insert(0, str(R / 'scripts'))
from resource_lease import lease, wait_memory

STATE = {'state': 'starting', 'pid': os.getpid(), 'jobs': {}, 'training': False}
STATE_PATH = A / 'profile_queue.json'
LOG_SUFFIX = ''


def write():
    STATE['updated_at'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    path = STATE_PATH
    tmp = path.with_suffix('.writing')
    tmp.write_text(json.dumps(STATE, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


async def profile(run, gpu):
    STATE['jobs'][run] = {'state': 'waiting_resources', 'physical_gpu': gpu}
    write()
    async with lease([gpu]) as handles:
        await wait_memory([gpu], 10000)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
            CUDA_HOME='/mnt/ojc/miniconda3/envs/4dgaussians', PYTHONNOUSERSITE='1',
            OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
            MPLCONFIGDIR=str(A / 'cache/matplotlib'), TORCH_HOME=str(A / 'cache/torch'))
        args = ['/mnt/ojc/miniconda3/envs/4dgaussians/bin/python', str(A / 'quality_profile.py'),
                '--run', str(A / 'runs' / run)]
        logpath = A / 'logs' / f'profile_{run}{LOG_SUFFIX}.log'
        with logpath.open('xb') as log:
            process = await asyncio.create_subprocess_exec(*args, cwd=A, env=environment,
                stdout=log, stderr=asyncio.subprocess.STDOUT, pass_fds=handles)
            STATE['jobs'][run].update(state='running', pid=process.pid, command=args,
                                      log=str(logpath), started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
            write()
            code = await process.wait()
        (A / 'logs' / f'profile_{run}{LOG_SUFFIX}.exit').write_text(str(code) + '\n')
        STATE['jobs'][run].update(state='complete' if code == 0 else 'failed', exit_code=code,
                                  ended_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
        write()
        if code:
            raise RuntimeError(f'{run} quality profile failed with exit{code};inspect{logpath}')


async def worker(first, second, gpu):
    await profile(first, gpu)
    if second is not None:
        await profile(second, gpu)


async def main():
    global STATE_PATH, LOG_SUFFIX
    parser = ArgumentParser()
    parser.add_argument('--historical-schema-retry', action='store_true')
    args = parser.parse_args()
    if args.historical_schema_retry:
        original = json.loads(STATE_PATH.read_text())
        assert original['state'] == 'failed'
        assert original['jobs']['walk063_recipe30k_v1']['exit_code'] == 1
        assert original['jobs']['0084_recipe30k_v1']['exit_code'] == 1
        STATE_PATH = A / 'profile_retry_queue.json'
        LOG_SUFFIX = '_schema_r2'
    assert not STATE_PATH.exists(), 'Do not restart a completed or running queue'
    write()
    if args.historical_schema_retry:
        results = await asyncio.gather(
            profile('walk063_recipe30k_v1', 6), profile('0084_recipe30k_v1', 4),
            return_exceptions=True)
    else:
        results = await asyncio.gather(
            worker('m02_61_recipe30k_v1', '0084_recipe30k_v1', 4),
            worker('r07_61_recipe30k_v1', 'walkturn0021_recipe30k_v1', 5),
            worker('walk063_recipe30k_v1', None, 6),
            worker('phone_recipe30k_v1', None, 7), return_exceptions=True)
    failures = [str(item) for item in results if isinstance(item, Exception)]
    STATE.update(state='requested_dense_profiles_and_ssim_supplements_complete' if not failures else 'failed',
                 failures=failures)
    write()
    if failures:
        raise RuntimeError('; '.join(failures))


if __name__ == '__main__':
    asyncio.run(main())
