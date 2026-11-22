"""One bounded asynchronous GPU job, cooperating with the existing Waymo leases."""
import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
LEASES = Path('/mnt/ojc/workplace1/4danyone/waymo/gpu_leases_20261019')


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--minimum-free-mib', type=int, default=6000)
    parser.add_argument('--preferred-gpu', type=int, choices=[4,5,6,7], default=4)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.name.replace('_','').isalnum() and args.command
    status = ROOT/'logs'/f'{args.name}_process.json'
    state = dict(state='waiting_gpu', controller_pid=os.getpid(), started=time.time(), command=args.command)
    def save():
        temp = status.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(status)
    save()
    lease = None
    while lease is None:
        query = await asyncio.create_subprocess_exec('nvidia-smi',
            '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits',
            stdout=asyncio.subprocess.PIPE)
        stdout, _ = await query.communicate()
        assert query.returncode == 0
        candidates = [line.split(', ') for line in stdout.decode().strip().splitlines()]
        for index, uuid, free in candidates:
            if int(index) not in (4,5,6,7) or int(free) < args.minimum_free_mib:
                continue
            handle = (LEASES/f'gpu{index}.lock').open('a+')
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            lease = handle
            state.update(physical_gpu=int(index), gpu_uuid=uuid)
            break
        if lease is None:
            # Queue on the same OS lock. Polling alone misses the tiny gap between
            # successive Waymo jobs and can starve indefinitely.
            choice = next(row for row in candidates if int(row[0]) == args.preferred_gpu)
            index, uuid, _ = choice
            handle = (LEASES/f'gpu{index}.lock').open('a+')
            state.update(state='queued_shared_lease', physical_gpu=int(index), gpu_uuid=uuid)
            save()
            await asyncio.to_thread(fcntl.flock, handle.fileno(), fcntl.LOCK_EX)
            check = await asyncio.create_subprocess_exec('nvidia-smi',f'--id={uuid}',
                '--query-gpu=memory.free','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE)
            out,_ = await check.communicate()
            assert check.returncode == 0
            if int(out.decode().strip()) >= args.minimum_free_mib:
                lease = handle
            else:
                handle.close()
                await asyncio.sleep(10)
    try:
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=state['gpu_uuid'], FD_EXPECTED_GPU_UUID=state['gpu_uuid'],
                   PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        with (ROOT/'logs'/f'{args.name}.log').open('a') as log:
            proc = await asyncio.create_subprocess_exec(*args.command, cwd=ROOT,
                env=env, stdout=log, stderr=asyncio.subprocess.STDOUT)
            state.update(state='running', pid=proc.pid, launched=time.time())
            save()
            code = await proc.wait()
        state.update(state='completed' if code==0 else 'failed', exit_code=code, finished=time.time())
        save()
        if code:
            raise SystemExit(code)
    finally:
        lease.close()


if __name__ == '__main__':
    asyncio.run(main())
