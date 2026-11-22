"""Finish active GPU stages, then replace scheduler at guarded stage boundaries."""
import asyncio
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo
import psutil

R=Path(__file__).resolve().parents[1];B=R.parent/'batch100'
PY='/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python'
def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()

async def main():
    request=json.loads((B/'handoff.json').read_text())
    parent=psutil.Process(request['old_controller_pid'])
    assert str(R/'scripts/batch100.py') in parent.cmdline()
    record={'started_at':now(),'old_controller_pid':parent.pid,'target':24,'state':'waiting_for_gpu_stage_to_finish'}
    write(B/'retarget24.json',record)
    while True:
        children=parent.children()
        def guarded(child):
            marker=B/'handoff_waiters'/f'{child.pid}.json'
            return marker.exists() and json.loads(marker.read_text())['parent']==parent.pid
        if children and all(map(guarded,children)):
            await asyncio.sleep(1)
            if all(map(guarded,parent.children())):break
        await asyncio.sleep(5)
    record.update(state='all_active_stages_finished',guard_pids=[c.pid for c in children],updated_at=now())
    write(B/'retarget24.json',record)
    # The old session owns only the scheduler and guarded, idle children here.
    stop=await asyncio.create_subprocess_exec('tmux','-L','fdwaymo','kill-session','-t','batch100')
    if await stop.wait():raise RuntimeError('Old scheduler session did not close')
    await asyncio.to_thread(parent.wait,10)
    # Guard children exit without running their stages and release the old lock.
    with (B/'controller.lock').open('a') as old_lock:
        await asyncio.to_thread(fcntl.flock,old_lock.fileno(),fcntl.LOCK_EX)
    request.update(state='complete',updated_at=now());write(B/'handoff.json',request)
    record.update(state='resuming_with_target24',updated_at=now());write(B/'retarget24.json',record)
    os.execv(PY,[PY,'-u',str(R/'scripts/batch100.py')])

if __name__=='__main__':
    with (B/'retarget.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        asyncio.run(main())
