"""Read-only GPU support diagnostics across the34 completed assets."""
import asyncio
import fcntl
import os
from baseline_queue import ROOT, LABELS, BRANCHES, gpu_info, save, now


async def main():
    path = ROOT / 'audit/support_queue.json'
    assert not path.exists()
    queue = asyncio.Queue()
    queue.put_nowait(('N04', 'grid4d_33k'))
    queue.put_nowait(('N11', 'grid4d_33k'))
    for label in LABELS:
        for branch in BRANCHES:
            queue.put_nowait((label, branch))
    state = dict(state='running', pid=os.getpid(), jobs={}, started_at=now())
    save(path, state)
    async def worker(index):
        with (ROOT / 'cache/gpu_locks' / f'gpu{index}.lock').open('a') as lock:
            while not queue.empty():
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                gpu = (await gpu_info())[index]
                if gpu['free_mib'] < 8000:
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    await asyncio.sleep(20)
                    continue
                try:
                    label, method = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                envname = 'grid4d' if method == 'grid4d_33k' else '4danyone-ftgs' if method.startswith('ftgs') else '4dgaussians'
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
                env.pop('PYTHONPATH', None)
                if method != 'grid4d_33k':
                    env['TORCH_EXTENSIONS_DIR'] = str(ROOT / 'cache/baseline_extensions')
                command = [f'/mnt/ojc/miniconda3/envs/{envname}/bin/python', '-u', str(ROOT / 'scripts/terminal_support.py'), '--label', label, '--method', method]
                key = f'{label}_{method}'; log = ROOT / 'logs' / f'support_{key}.log'
                with log.open('x') as output:
                    child = await asyncio.create_subprocess_exec(*command, cwd=ROOT, env=env, stdout=output, stderr=asyncio.subprocess.STDOUT)
                    record = dict(state='running', pid=child.pid, gpu=gpu, command=command, log=str(log), started_at=now())
                    state['jobs'][key] = record; save(path, state)
                    code = await child.wait()
                record.update(state='complete' if code == 0 else 'failed', exit_code=code, ended_at=now()); save(path, state)
                queue.task_done(); fcntl.flock(lock, fcntl.LOCK_UN)
                if code:
                    return
    await asyncio.gather(*(worker(i) for i in [6, 7, 0, 1, 2, 3, 4, 5]))
    state.update(state='complete' if queue.empty() and all(r['state'] == 'complete' for r in state['jobs'].values()) else 'partial_requires_review', pending=queue.qsize(), ended_at=now())
    save(path, state)


if __name__ == '__main__':
    asyncio.run(main())
