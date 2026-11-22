"""Bounded asynchronous replay/evaluation; publish only the complete four-case set."""
import asyncio
import fcntl
import json
import os
from PIL import Image
from baseline_queue import ROOT, gpu_info, save, now


async def main():
    path = ROOT / 'mechanism/static/EVALUATION_STATUS.json'
    assert not path.exists()
    state = dict(state='running', jobs={}, started_at=now(), pid=os.getpid())
    save(path, state)
    locks = {6: asyncio.Lock(), 7: asyncio.Lock()}

    async def job(case, index):
        async with locks[index]:
            with (ROOT / 'cache/gpu_locks' / f'gpu{index}.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                gpu = (await gpu_info())[index]
                assert gpu['free_mib'] > 4000
                command = ['/mnt/ojc/miniconda3/envs/grid4d/bin/python', '-u', str(ROOT / 'scripts/evaluate_static.py'), '--case', case]
                log = ROOT / 'logs' / f'static_evaluate_{case}.log'
                with log.open('x') as out:
                    proc = await asyncio.create_subprocess_exec(*command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid']), stdout=out, stderr=asyncio.subprocess.STDOUT)
                    record = dict(state='running', pid=proc.pid, gpu=gpu, command=command, log=str(log), started_at=now())
                    state['jobs'][case] = record
                    save(path, state)
                    code = await proc.wait()
                record.update(state='complete' if code == 0 else 'failed', exit_code=code, ended_at=now())
                save(path, state)

    await asyncio.gather(job('N04_t000', 6), job('N04_t060', 6), job('N11_t000', 7), job('N11_t060', 7))
    assert all(r['state'] == 'complete' for r in state['jobs'].values()), state
    cases = ['N04_t000', 'N04_t060', 'N11_t000', 'N11_t060']
    stage = ROOT / 'tmp/static_evaluation'
    reports = {case: json.loads((stage / case / 'metrics.json').read_text()) for case in cases}
    save(stage / 'summary.json', dict(state='complete', cases=reports, reference='Generated-heldout fixed-time diagnostic, not real-ground-truth or matched-compute comparison'))
    contact = Image.new('RGB', (1408, 672 * 4))
    for i, case in enumerate(cases):
        with Image.open(stage / case / 'view00.png') as picture:
            contact.paste(picture, (0, 672 * i))
    contact.save(stage / 'comparison.png')
    destination = ROOT / 'mechanism/static/results'
    assert not destination.exists()
    stage.rename(destination)
    state.update(state='complete', ended_at=now(), results=str(destination))
    save(path, state)


if __name__ == '__main__':
    asyncio.run(main())
