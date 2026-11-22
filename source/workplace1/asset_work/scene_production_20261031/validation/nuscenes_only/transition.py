"""One-time verified boundary restart; never interrupt generation children."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,'/mnt/ojc/DriveLab')
from asset_extraction.io_utils import write_json


def main():
    work=Path(__file__).resolve().parent
    with (work/'transition.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=int(sys.argv[1]) if len(sys.argv)>1 else 387476
        command=Path('/proc')/str(old)/'cmdline'
        write_json(work/'transition.json',dict(status='waiting_for_current_chunks',old_pid=old,updated=time.time()))
        while command.exists() and command.read_bytes():
            cmd=command.read_bytes().replace(b'\0',b' ').decode()
            if str(ROOT/'run.py') not in cmd or '--maintain' in cmd:
                raise RuntimeError('Old PID identity changed; refusing to manage an unrelated process')
            time.sleep(5)
        control=json.loads((ROOT/'dataset_control.json').read_text())
        if control['nuscenes']!='running' or control['waymo']!='paused':
            raise RuntimeError('Dataset preference changed; refusing stale transition')
        waymo=json.loads((ROOT/'waymo_status.json').read_text())
        nu=json.loads((ROOT/'nuscenes_status.json').read_text())
        if waymo['status'] not in {'drained','paused_by_user'} or nu['status']!='drained':
            raise RuntimeError('Old supervisor did not complete a clean boundary drain')
        if waymo['current_chunk']!=control['waymo_last_allowed_chunk']:
            raise RuntimeError('Waymo progressed beyond the authorized last chunk')
        receipt=Path(waymo['current_chunk'])/'completed.json'
        if not receipt.is_file() or str(receipt.parent) not in waymo['chunks']:
            raise RuntimeError('Final Waymo chunk is not fully committed')
        args=[sys.executable,str(ROOT/'run.py')]
        logfile=ROOT/'supervisor_nuscenes_only.log'
        with logfile.open('ab') as out:
            process=subprocess.Popen(args,cwd='/mnt/ojc/DriveLab',stdin=subprocess.DEVNULL,
                stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        previous=json.loads((ROOT/'background_processes.json').read_text())
        write_json(work/'previous_background_processes.json',previous)
        records=[dict(name='supervisor',pid=process.pid,args=args,started=time.time(),log=logfile.name)]
        records.extend(x for x in previous if x['name']!='supervisor')
        write_json(ROOT/'background_processes.json',records)
        time.sleep(2)
        if process.poll() is not None:
            raise RuntimeError(f'New supervisor exited {process.returncode}; inspect {logfile}')
        write_json(work/'transition.json',dict(status='new_supervisor_running',pid=process.pid,
            old_pid=old,waymo_last_chunk=waymo['current_chunk'],waymo_queries=waymo['queries'],
            waymo_kept=waymo['kept'],updated=time.time()))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write_json(Path(__file__).with_name('transition_error.json'),dict(error=repr(exc),updated=time.time()))
        raise
