"""Finish active chunks, then replace the supervisor with the non-car priority policy."""
from pathlib import Path
import os,json,time,urllib.request,fcntl,sys
ROOT=Path('/mnt/ojc/workplace1/asset_work/scene_production_20261031')
OUT=Path(__file__).parent
PYTHON='/mnt/ojc/miniconda3/envs/vllm_vlm/bin/python'
old_pid=int(sys.argv[1]);old_start=sys.argv[2]
def save(p,d):
 tmp=p.with_name(p.name+'.handoff.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def same_old():
 try:
  data=(Path('/proc')/str(old_pid)/'stat').read_text().split(') ',1)[1].split()
  return data[0]!='Z' and data[19]==old_start
 except FileNotFoundError:return False
save(OUT/'handoff_state.json',dict(status='waiting_for_active_chunks',old_pid=old_pid,pid=os.getpid(),created=time.time(),threshold_GiB=10))
while same_old():time.sleep(5)
control=json.loads((ROOT/'dataset_control.json').read_text())
if not any(control.get(ds)=='running' for ds in ['nuscenes','waymo']):
 save(OUT/'handoff_state.json',dict(status='cancelled_by_user_pause',time=time.time()));raise SystemExit(0)
while True:
 try:
  with urllib.request.urlopen('http://127.0.0.1:8082/health',timeout=8) as response:health=json.load(response)
  if len(health['backends'])==2 and all(x['healthy'] for x in health['backends'].values()):break
 except Exception as exc:print('Waiting for Qwen health:',repr(exc),flush=True)
 time.sleep(5)
with (ROOT/'supervisor.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
sys.path.insert(0,str(ROOT))
import runtime
assert runtime.minimum_free_gib()==10
assert control.get('vehicle_priority')=='noncar_three_to_one_within_scene_v1'
records=json.loads((ROOT/'background_processes.json').read_text());records=[r for r in records if r['name'] not in ['supervisor','supervisor_handoff']]
records.insert(0,dict(name='supervisor',pid=os.getpid(),args=[PYTHON,str(ROOT/'run.py')],started=time.time(),status='running',log='supervisor_noncar_priority_20261103.log'))
save(ROOT/'background_processes.json',records)
save(OUT/'handoff_state.json',dict(status='exec_new_supervisor',pid=os.getpid(),old_pid=old_pid,minimum_free_gib=runtime.minimum_free_gib(),time=time.time()))
print('Current chunks completed; starting non-car priority supervisor with dynamic 10 GiB threshold',os.getpid(),flush=True)
os.execv(PYTHON,[PYTHON,str(ROOT/'run.py')])
