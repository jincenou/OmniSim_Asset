import json,os,signal,subprocess,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PYTHON='/mnt/ojc/miniconda3/envs/vllm_vlm/bin/python'
children=[]
def launch(name,args):
 log=(ROOT/'logs'/f'{name}.log').open('ab',buffering=0)
 p=subprocess.Popen([PYTHON,'-u',*args],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,ASSET_VLM_MEMORY_UTILIZATION='0.65'))
 children.append((p,log));return p
try:
 for gpu,port in [(0,8085),(3,8086),(4,8083),(7,8084)]:launch(f'qwen_extra_gpu{gpu}',[str(ROOT/'screening_vlm.py'),'--replica',str(gpu),str(port)])
 launch('screening_dispatcher',[str(ROOT/'screening_vlm.py'),'--port','8087'])
 deadline=time.time()+900
 while time.time()<deadline:
  if any(p.poll() is not None for p,_ in children):raise RuntimeError('screening service exited')
  try:
   d=json.loads((ROOT/'screening_cluster/dispatcher_status.json').read_text())
   if len(d['backends'])==4 and all(v['healthy'] for v in d['backends'].values()):break
  except Exception:pass
  time.sleep(3)
 else:raise TimeoutError('four Qwen replicas not healthy')
 (ROOT/'screening_status.json').write_text(json.dumps(dict(status='screening_four_replicas',gpus=[0,3,4,7],port=8087,ready_at=time.time()),indent=2))
 while not (ROOT/'STOP_SCREENING').exists():
  if any(p.poll() is not None for p,_ in children):raise RuntimeError('Qwen screening backend exited')
  time.sleep(5)

finally:
 for p,_ in children:
  if p.poll() is None:p.terminate()
 deadline=time.time()+25
 for p,h in children:
  try:p.wait(timeout=max(1,deadline-time.time()))
  except subprocess.TimeoutExpired:p.kill();p.wait()
  h.close()
