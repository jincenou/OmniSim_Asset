"""Two bounded asynchronous, read-only GPU audits with shared project locks."""
import asyncio
import fcntl
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

NEW = Path(__file__).resolve().parents[1]

async def run(label, index, uuid):
    lock = (NEW.parent/'cache/gpu_locks'/f'gpu{index}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = dict(label=label, gpu=index, uuid=uuid, started_at=datetime.now(timezone.utc).isoformat())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid)
    with (NEW/'logs'/f'{label}_cutoff_audit.log').open('x') as log:
        process = await asyncio.create_subprocess_exec(sys.executable, str(NEW/'scripts/audit_cutoff.py'), '--label', label, env=env, stdout=log, stderr=asyncio.subprocess.STDOUT)
        status['pid'] = process.pid
        status['exit_code'] = await process.wait()
    status['ended_at'] = datetime.now(timezone.utc).isoformat()
    lock.close()
    return status

async def main():
    result = await asyncio.gather(run('N04', 6, 'GPU-cd4b2342-fba4-2b42-beb9-60b698debc2d'), run('N11', 7, 'GPU-32040e1a-df6d-0776-5420-fae47bc11248'))
    (NEW/'research/cutoff_processes.json').write_text(json.dumps(result, indent=2)+'\n')
    assert all(row['exit_code'] == 0 for row in result), result

asyncio.run(main())
