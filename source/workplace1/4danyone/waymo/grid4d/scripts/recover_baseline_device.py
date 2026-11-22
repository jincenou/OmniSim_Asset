"""One explicit retry of four pre-render device-property failures."""
import asyncio
import fcntl
import json
import os
from baseline_queue import ROOT, gpu_info, now, save


async def main():
    state = dict(state='running', reason='torch2.3 has no properties.uuid; use actual CUDA context UUID via driver API for both runtimes', jobs={})
    gpu = await gpu_info()
    async def run(label, branch, index):
        key = f'{label}_{branch}'
        prior = json.loads((ROOT / 'logs' / f'baseline_{key}.json').read_text())
        assert prior['exit_code'] == 1
        assert "object has no attribute 'uuid'" in (ROOT / 'logs' / f'baseline_{key}.log').read_text()
        assert not (ROOT / 'tmp/baseline_raw' / label / branch).exists()
        assert gpu[index]['free_mib'] >= 10000
        with (ROOT / 'cache/gpu_locks' / f'gpu{index}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu[index]['uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       TORCH_HOME=str(ROOT / 'cache/torch'), MPLCONFIGDIR=str(ROOT / 'cache/matplotlib'),
                       TORCH_EXTENSIONS_DIR=str(ROOT / 'cache/baseline_extensions'), CUDA_CACHE_PATH=str(ROOT / 'cache/cuda'),
                       TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache/xdg'))
            env.pop('PYTHONPATH', None)
            command = ['/mnt/ojc/miniconda3/envs/4dgaussians/bin/python', '-u', str(ROOT / 'scripts/render_baseline.py'), '--label', label, '--branch', branch]
            log = ROOT / 'logs' / f'baseline_{key}_retry1.log'
            with log.open('x') as output:
                process = await asyncio.create_subprocess_exec(*command, cwd=ROOT, env=env, stdout=output, stderr=asyncio.subprocess.STDOUT)
                record = dict(state='running', pid=process.pid, gpu=gpu[index], command=command, started_at=now(), log=str(log), original_failure=prior)
                state['jobs'][key] = record
                save(ROOT / 'audit/render_recovery.json', state)
                code = await process.wait()
            record.update(state='complete' if code == 0 else 'failed', exit_code=code, ended_at=now())
            save(ROOT / 'logs' / f'baseline_{key}_retry1.json', record)
            save(ROOT / 'audit/render_recovery.json', state)
    await asyncio.gather(run('N04', '4dgs_grid61', 0), run('N04', '4dgs_grid25', 1),
                         run('N11', '4dgs_grid61', 4), run('N11', '4dgs_grid25', 5))
    state.update(state='complete' if all(v['state'] == 'complete' for v in state['jobs'].values()) else 'failed_requires_review', ended_at=now())
    save(ROOT / 'audit/render_recovery.json', state)


if __name__ == '__main__':
    asyncio.run(main())
