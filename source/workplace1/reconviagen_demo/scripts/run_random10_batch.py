"""Eight shared-GPU workers with disk reservations and explicit task status."""
from pathlib import Path
import json,subprocess,os,time,threading,shutil
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
matrix=ROOT/'inputs/random10_experiment_matrix.json';tasks=json.loads(matrix.read_text())
tasks.sort(key=lambda t:(['B0','R','F','H','B0_real1','B0_real4'].index(t['variant']),t['case_id']))
lock=threading.Lock();running=set();reserve=350*1024**2
def save():matrix.write_text(json.dumps(tasks,indent=2))
def pending_download():
 folder=ROOT/'cache/timm--vit_huge_plus_patch16_dinov3_qkvb.lvd1689m'
 if (folder/'model.safetensors').exists():return 0
 chunks=list(folder.rglob('*.incomplete'))
 return max(0,3362418152-sum(f.stat().st_size for f in chunks))
def worker(gpu):
 while True:
  with lock:
   pending=[t for t in tasks if t['status'] in ('inputs_frozen','waiting_disk')]
   if not pending:return
   free=shutil.disk_usage(ROOT).free;needed=2*1024**3+pending_download()+reserve*(len(running)+1)
   if free<needed:
    for t in pending:t['status']='waiting_disk'
    save();task=None
   else:
    task=pending[0];task.update(status='running',gpu=gpu,started_unix=time.time());running.add(gpu);save()
  if task is None:time.sleep(20);continue
  name=task['case_id']+'_'+task['variant'];log=ROOT/'logs'/(name+'.log')
  print('START',gpu,name,flush=True)
  env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
  with log.open('w') as f:
   result=subprocess.run(['python',str(ROOT/'scripts/run_variant.py'),'--case',task['case_id'],'--variant',task['variant'],'--gpu',str(gpu),'--seed',str(task['seed'])],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
  with lock:
   task.update(status='rendered_heading_review_pending' if result.returncode==0 else 'failed',exit_code=result.returncode,finished_unix=time.time(),log=str(log));running.remove(gpu);save()
  print('END',gpu,name,result.returncode,flush=True)
with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(worker,range(8)))
print('ALL TASKS TERMINAL',flush=True)
