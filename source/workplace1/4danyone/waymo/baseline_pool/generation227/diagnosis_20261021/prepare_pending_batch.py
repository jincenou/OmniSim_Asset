"""Prepare remaining diagnostic evidence in one bounded CPU batch."""
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

R = Path(__file__).resolve().parent
STATE = {}


def save():
    STATE['updated_at'] = time.time()
    p = R / 'pending_batch_status.json'
    tmp = p.with_suffix('.writing')
    tmp.write_text(json.dumps(STATE, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


async def main():
    lock = (R / 'pending_batch.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((R / 'pending_batch_plan.json').read_text())
    assert hashlib.sha256((R / 'evidence.py').read_bytes()).hexdigest() == plan['extractor_sha256']
    assert hashlib.sha256((R / 'classification.json').read_bytes()).hexdigest() == plan['classification_sha256']
    STATE.update(state='running', pid=os.getpid(), started_at=time.time(), jobs={})
    save()
    sem = asyncio.Semaphore(plan['cpu_concurrency'])

    async def one(index):
        async with sem:
            result = R / f'evidence_{index:03d}.json'
            image = R / f'evidence_{index:03d}.jpg'
            if result.exists():
                receipt = json.loads(result.read_text())
                assert receipt['index'] == index and image.is_file()
                STATE['jobs'][str(index)] = dict(state='existing_evidence_not_overwritten')
                save()
                return
            child = await asyncio.create_subprocess_exec(
                sys.executable, str(R / 'evidence.py'), str(index),
                env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                         PYTHONDONTWRITEBYTECODE='1'))
            STATE['jobs'][str(index)] = dict(state='running', pid=child.pid)
            save()
            code = await child.wait()
            STATE['jobs'][str(index)].update(exit_code=code, finished_at=time.time(),
                                              state='failed' if code else 'evidence_prepared_not_reviewed')
            if not code:
                receipt = json.loads(result.read_text())
                assert receipt['index'] == index and image.is_file()
            save()

    await asyncio.gather(*(one(i) for i in plan['indices_to_prepare']))
    failed = [k for k, row in STATE['jobs'].items() if row['state'] == 'failed']
    STATE.update(state='evidence_prepared_visual_review_pending' if not failed else 'partial_failure',
                 failed_indices=failed, finished_at=time.time())
    save()


if __name__ == '__main__':
    asyncio.run(main())
