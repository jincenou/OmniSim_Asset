"""Drain currently running subprocesses, reconcile exits, then resume with target 24."""
import asyncio
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import signal
from zoneinfo import ZoneInfo
import psutil

R=Path(__file__).resolve().parents[1];B=R.parent/'batch100'
PY='/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python'


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def exit_status(pid):
    path=Path(f'/proc/{pid}/stat')
    if not path.exists():raise RuntimeError(f'Lost unreaped stage PID {pid}; manual log verification required')
    fields=path.read_text().rsplit(')',1)[1].split()
    if fields[0]!='Z':return None
    return os.waitstatus_to_exitcode(int(fields[49]))


async def main():
    controller=json.loads((B/'controller.json').read_text());parent=psutil.Process(controller['pid'])
    assert str(R/'scripts/batch100.py') in parent.cmdline(),parent.cmdline()
    original_status=parent.status();assert original_status!=psutil.STATUS_STOPPED
    record={'started_at':now(),'old_controller_pid':parent.pid,'old_controller_created':parent.create_time(),
            'target':24,'state':'draining_existing_stages','stages':[]}
    parent.send_signal(signal.SIGSTOP)
    try:
        deadline=asyncio.get_running_loop().time()+3
        while parent.status()!=psutil.STATUS_STOPPED:
            if asyncio.get_running_loop().time()>deadline:
                raise RuntimeError('Controller did not stop; refusing unsafe migration')
            await asyncio.sleep(.05)
        # With admissions suspended, subprocess stdout identifies its immutable stage log.
        children=parent.children()
        jobs=[json.loads(path.read_text()) for path in (B/'jobs').glob('*.json')]
        for child in children:
            try:stdout=Path(f'/proc/{child.pid}/fd/1').resolve(strict=True)
            except FileNotFoundError:
                stdout=None
            if stdout is None:
                owners=[job for job in jobs if job.get('pid')==child.pid]
                if len(owners)==1:stdout=Path(owners[0]['log'])
            match=re.fullmatch(r'(batch_\d+)-([a-z0-9_]+)-(\d+)\.log',stdout.name) if stdout else None
            if match:
                assert stdout.parent==B/'logs',stdout
                name,stage,attempt=match.groups()
                record['stages'].append({'pid':child.pid,'name':name,'stage':stage,'attempt':int(attempt),'log':str(stdout)})
            elif child.status()!=psutil.STATUS_ZOMBIE:
                # The only non-stage child is the bounded read-only resource query.
                assert Path(child.exe()).name=='nvidia-smi',child.cmdline()
        write(B/'retarget24.json',record)
        while any(exit_status(child.pid) is None for child in children):
            await asyncio.sleep(5)
        for stage in record['stages']:
            code=exit_status(stage['pid']);stage['exit_code']=code
            path=B/'jobs'/f"{stage['name']}.json";job=json.loads(path.read_text())
            assert job['stage']==stage['stage'],(job['stage'],stage)
            Path(stage['log']).with_suffix('.exit').write_text(str(code)+'\n')
            job.pop('pid',None);job['updated_at']=now()
            if code==0:
                if stage['stage'] not in job['done']:job['done'].append(stage['stage'])
                job['state']='running'
            else:job.update(state='failed',error=f"{stage['stage']} exit {code}; {stage['log']}")
            job['reconciled_by']='retarget24; actual kernel child exit status'
            write(path,job)
        record.update(state='stages_finished_restarting_controller',updated_at=now());write(B/'retarget24.json',record)
        # All subprocesses are already terminal; only the stopped old scheduler is removed.
        assert all(child.status()==psutil.STATUS_ZOMBIE for child in parent.children())
        parent.kill();await asyncio.to_thread(parent.wait,10)
    except BaseException:
        if parent.is_running() and parent.status()==psutil.STATUS_STOPPED:parent.send_signal(signal.SIGCONT)
        record.update(state='migration_failed_old_controller_resumed',updated_at=now());write(B/'retarget24.json',record)
        raise
    record.update(state='resuming_with_target24',updated_at=now());write(B/'retarget24.json',record)
    os.execv(PY,[PY,'-u',str(R/'scripts/batch100.py')])


if __name__=='__main__':
    with (B/'retarget.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        asyncio.run(main())
