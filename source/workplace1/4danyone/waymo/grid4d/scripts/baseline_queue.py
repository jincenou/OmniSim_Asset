"""Bounded async workers for useful read-only GPU diagnostics, not new training."""
import argparse
import asyncio
import fcntl
import json
import os
import shutil
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'random8/r20261015_113332'
LABELS = ['N04', 'N11', 'N01', 'N07', 'N02', 'N03', 'N06', 'N08']
BRANCHES = ['4dgs_grid61', '4dgs_grid25', 'ftgs_opt15k', 'ftgs_base5k']


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    tmp = path.with_suffix('.writing')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


async def gpu_info():
    process = await asyncio.create_subprocess_exec('nvidia-smi', '--query-gpu=index,uuid,memory.free',
        '--format=csv,noheader,nounits', stdout=asyncio.subprocess.PIPE)
    data, _ = await process.communicate()
    assert process.returncode == 0
    return {int(v[0]): dict(index=int(v[0]), uuid=v[1].strip(), free_mib=int(v[2]))
            for line in data.decode().splitlines() if (v := line.split(','))}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='initial')
    parser.add_argument('--gpus', nargs='+', type=int, default=list(range(6)))
    args = parser.parse_args()
    assert args.run_id.replace('_', '').isalnum()
    assert len(set(args.gpus)) == len(args.gpus) and all(0 <= i < 8 for i in args.gpus)
    suffix = '' if args.run_id == 'initial' else '_' + args.run_id
    state_path = ROOT / 'audit' / f'render_queue{suffix}.json'
    assert not state_path.exists(), 'Use a new reviewed run ID; preserve previous queue history'
    queue = asyncio.Queue()
    for label in LABELS:
        for branch in BRANCHES:
            if not (ROOT / 'tmp/baseline_raw' / label / branch / 'complete.json').exists():
                queue.put_nowait((label, branch))
    state = dict(state='running', pid=os.getpid(), started_at=now(), jobs={}, gpu_pool=args.gpus)
    lockdir = ROOT / 'cache/gpu_locks'
    lockdir.mkdir(parents=True, exist_ok=True)
    async def worker(index):
        local = (lockdir / f'gpu{index}.lock').open('a')
        try:
            while not queue.empty():
                try:
                    fcntl.flock(local, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    await asyncio.sleep(15)
                    continue
                gpu = (await gpu_info())[index]
                if gpu['free_mib'] < 10000 or shutil.disk_usage(ROOT).free < 40 * 2**30:
                    fcntl.flock(local, fcntl.LOCK_UN)
                    await asyncio.sleep(30)
                    continue
                try:
                    label, branch = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                key = f'{label}_{branch}'
                log = ROOT / 'logs' / f'baseline_{key}{suffix}.log'
                assert not log.exists(), 'Review previous job receipt; no blind restart'
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                    TORCH_HOME=str(ROOT / 'cache/torch'), MPLCONFIGDIR=str(ROOT / 'cache/matplotlib'),
                    TORCH_EXTENSIONS_DIR=str(ROOT / 'cache/baseline_extensions'), CUDA_CACHE_PATH=str(ROOT / 'cache/cuda'),
                    TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache/xdg'))
                environment.pop('PYTHONPATH', None)
                envname = '4danyone-ftgs' if branch.startswith('ftgs') else '4dgaussians'
                command = [f'/mnt/ojc/miniconda3/envs/{envname}/bin/python', '-u', str(ROOT / 'scripts/render_baseline.py'), '--label', label, '--branch', branch]
                with log.open('x') as output:
                    child = await asyncio.create_subprocess_exec(*command, cwd=ROOT, env=environment, stdout=output, stderr=asyncio.subprocess.STDOUT)
                    record = dict(state='running', gpu=gpu, command=command, pid=child.pid, started_at=now(), log=str(log))
                    state['jobs'][key] = record
                    save(state_path, state)
                    code = await child.wait()
                record.update(state='complete' if code == 0 else 'failed', exit_code=code, ended_at=now())
                save(state_path, state)
                save(ROOT / 'logs' / f'baseline_{key}{suffix}.json', record)
                queue.task_done()
                fcntl.flock(local, fcntl.LOCK_UN)
                if code:
                    # Preserve failure; other independent workers continue their assigned methods.
                    print(json.dumps(dict(job=key, state='failed', log=str(log))), flush=True)
                    return
        finally:
            local.close()
    await asyncio.gather(*(worker(index) for index in args.gpus))
    state.update(state='complete' if queue.empty() and all(j['state'] == 'complete' for j in state['jobs'].values()) else 'partial_requires_review', ended_at=now(), pending=queue.qsize())
    save(state_path, state)


if __name__ == '__main__':
    asyncio.run(main())
