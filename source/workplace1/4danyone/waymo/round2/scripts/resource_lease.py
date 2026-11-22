"""Shared phase lock plus per-device exclusive leases; four-card stages use EX."""
import asyncio
from contextlib import asynccontextmanager
import fcntl
from pathlib import Path

ROUND=Path(__file__).resolve().parents[1]


@asynccontextmanager
async def lease(ids):
    if not ids or not set(ids)<=set(range(4,8)):raise ValueError('Unauthorized device lease')
    directory=ROUND/'gpu_locks';directory.mkdir(exist_ok=True)
    handles=[(ROUND/'controller.lock').open('a'),(ROUND.parent/'batch100/controller.lock').open('a')]
    handles.extend((directory/f'gpu{gpu}.lock').open('a') for gpu in sorted(ids))
    try:
        while True:
            acquired=[]
            try:
                for index,handle in enumerate(handles):
                    fcntl.flock(handle, (fcntl.LOCK_SH if index<2 else fcntl.LOCK_EX)|fcntl.LOCK_NB)
                    acquired.append(handle)
            except BlockingIOError:
                for handle in acquired:fcntl.flock(handle,fcntl.LOCK_UN)
                await asyncio.sleep(30)
            else:break
        yield tuple(handle.fileno() for handle in handles)
    finally:
        for handle in handles:handle.close()


async def wait_memory(ids,minimum):
    while True:
        process=await asyncio.create_subprocess_exec('nvidia-smi','--id='+','.join(map(str,ids)),
            '--query-gpu=memory.free','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE)
        data,_=await process.communicate()
        if process.returncode:raise RuntimeError('GPU query failed')
        if min(map(int,data.splitlines()))>=minimum:return
        await asyncio.sleep(60)
