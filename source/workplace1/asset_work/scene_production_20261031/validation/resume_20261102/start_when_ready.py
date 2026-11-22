"""One-shot background resume: wait for both Qwen replicas, then exec supervisor."""
import json,os,time,urllib.request,fcntl
from pathlib import Path
ROOT=Path('/mnt/ojc/workplace1/asset_work/scene_production_20261031')
PYTHON='/mnt/ojc/miniconda3/envs/vllm_vlm/bin/python'
def save(path,data):
 t=path.with_suffix('.resume.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(path)
while True:
 if json.loads((ROOT/'dataset_control.json').read_text())['nuscenes']!='running':
  raise SystemExit('Resume cancelled by dataset control')
 try:
  with urllib.request.urlopen('http://127.0.0.1:8082/health',timeout=5) as response:health=json.load(response)
  if len(health['backends'])==2 and all(v['healthy'] for v in health['backends'].values()):break
 except Exception as exc:print('Waiting for Qwen:',str(exc),flush=True)
 time.sleep(5)
with (ROOT/'supervisor.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
records=json.loads((ROOT/'background_processes.json').read_text())
records=[r for r in records if r['name']!='supervisor']
records.insert(0,dict(name='supervisor',pid=os.getpid(),args=[PYTHON,str(ROOT/'run.py')],started=time.time(),status='running',log='supervisor_resume_20261102.log'))
save(ROOT/'background_processes.json',records)
v=json.loads((ROOT/'vlm_process.json').read_text());v.update(status='running',health_verified_at=time.time());save(ROOT/'vlm_process.json',v)
save(ROOT/'validation/resume_20261102/health_before_supervisor.json',health)
print('Both Qwen replicas healthy; exec production supervisor',os.getpid(),flush=True)
os.execv(PYTHON,[PYTHON,str(ROOT/'run.py')])
