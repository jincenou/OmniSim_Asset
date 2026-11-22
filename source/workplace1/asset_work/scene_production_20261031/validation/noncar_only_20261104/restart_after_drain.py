import os,sys,time,json,signal,fcntl
from pathlib import Path
r=Path('/mnt/ojc/workplace1/asset_work/scene_production_20261031');w=Path(__file__).parent
lock=(w/'transition.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
pid=621478;proc=Path('/proc')/str(pid)
def identity():
 try:return (proc/'stat').read_text().rsplit(')',1)[1].split()[19]
 except FileNotFoundError:return None
start=identity()
if start is not None:
 cmd=(proc/'cmdline').read_bytes().split(b'\0');assert str(r/'run.py').encode() in cmd and b'--maintain' not in cmd
 os.kill(pid,signal.SIGUSR1)
(w/'transition.json').write_text(json.dumps(dict(status='waiting_for_existing_chunks',old_pid=pid,old_start=start,watcher_pid=os.getpid(),time=time.time())))
while start is not None and identity()==start:time.sleep(5)
(w/'transition.json').write_text(json.dumps(dict(status='launching_noncar_supervisor',pid=os.getpid(),time=time.time())))
os.chdir(r)
os.execv(sys.executable,[sys.executable,'-u',str(r/'run.py')])
