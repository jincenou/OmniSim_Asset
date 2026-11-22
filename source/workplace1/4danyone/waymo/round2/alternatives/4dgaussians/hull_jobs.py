"""Four bounded async silhouette-feasibility diagnostics, no training."""
import asyncio
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


def write():
    STATE['updated_at'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    path = A / 'hull_queue.json'
    tmp = path.with_suffix('.writing')
    tmp.write_text(json.dumps(STATE, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


async def diagnostic(run, gpu):
    STATE['jobs'][run] = {'state': 'waiting_resources', 'physical_gpu': gpu}
    write()
    async with lease([gpu]) as handles:
        await wait_memory([gpu], 10000)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
            PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
        command = ['/mnt/ojc/miniconda3/envs/4dgaussians/bin/python', str(A / 'hull_consistency.py'),
                   '--run', str(A / 'runs' / run)]
        path = A / 'logs' / f'hull_{run}.log'
        with path.open('xb') as log:
            process = await asyncio.create_subprocess_exec(*command, cwd=A, env=environment,
                stdout=log, stderr=asyncio.subprocess.STDOUT, pass_fds=handles)
            STATE['jobs'][run].update(state='running', pid=process.pid, command=command,
                log=str(path), started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
            write()
            code = await process.wait()
        (A / 'logs' / f'hull_{run}.exit').write_text(str(code) + '\n')
        STATE['jobs'][run].update(state='complete' if code == 0 else 'failed', exit_code=code,
            ended_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
        write()
        if code:
            raise RuntimeError(f'{run} hull diagnostic failed with exit{code};inspect{path}')


async def main():
    assert not (A / 'hull_queue.json').exists(), 'Do not restart completed or running queue'
    write()
    result = await asyncio.gather(diagnostic('m02_61_recipe30k_v1', 4),
        diagnostic('r07_61_recipe30k_v1', 5), diagnostic('walk063_recipe30k_v1', 6),
        diagnostic('phone_recipe30k_v1', 7), return_exceptions=True)
    failures = [str(item) for item in result if isinstance(item, Exception)]
    STATE.update(state='four_two_resolution_silhouette_diagnostics_complete' if not failures else 'failed', failures=failures)
    write()
    if failures:
        raise RuntimeError('; '.join(failures))


if __name__ == '__main__':
    asyncio.run(main())
