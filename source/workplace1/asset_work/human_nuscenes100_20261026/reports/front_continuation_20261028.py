"""One-shot continuation of the already admitted pool after existing front queues exit."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/ojc/workplace1/asset_work/human_nuscenes100_20261026')
REPORT=ROOT/'reports/front_continuation_20261028.json'
def read(p):return json.loads(p.read_text())
def save(r):
 p=REPORT.with_suffix('.writing');p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');os.replace(p,REPORT)
def current(pid,start):
 try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]==start
 except FileNotFoundError:return False
r=read(REPORT);r.update(pid=os.getpid(),state='waiting_current_front_queues');save(r)
while any(current(x['pid'],x['start']) for x in r['watched']):
 if read(ROOT/'PRODUCTION_PAUSE.json').get('active_gpus')!=list(range(8)):
  r.update(state='cancelled_gpu_policy_changed');save(r);raise SystemExit
 time.sleep(30)
policy=read(ROOT/'PRODUCTION_PAUSE.json')
assert not policy['paused'] and policy['active_gpus']==list(range(8))
production=read(ROOT/'PRODUCTION_INPUTS.json');cases=[];withheld=[]
for identity in r['approved_cases']:
 assert identity in production['ids']
 if (Path('/mnt/ojc/workplace1/asset_library/human')/identity/'package_complete.json').exists():continue
 p=ROOT/'jobs'/f'{identity}.json';job=read(p) if p.exists() else {}
 if job.get('state')=='failed_or_needs_review':withheld.append(identity);continue
 cases.append(identity)
r.update(withheld_failures=withheld,cases=cases)
if not cases:r.update(state='no_remaining_work');save(r);raise SystemExit
command=['/mnt/ojc/miniconda3/envs/4danyone/bin/python','-u','/mnt/ojc/workplace1/human/run.py','batch','run','--cases',*cases,'--gpus','0','1','2','3','--queue-name','front_continuation','--min-free-mib','27000']
child=subprocess.Popen(command,cwd='/mnt/ojc/workplace1/human')
r.update(state='running_front_continuation',child_pid=child.pid,command=command,started_at=time.time());save(r)
exit_code=child.wait();r.update(state='complete' if exit_code==0 else 'needs_review',exit_code=exit_code,finished_at=time.time());save(r)
