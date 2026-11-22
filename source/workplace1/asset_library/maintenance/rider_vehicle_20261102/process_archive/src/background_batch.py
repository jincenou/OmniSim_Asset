import subprocess,json,sys,time,concurrent.futures
from pathlib import Path
W=Path(__file__).resolve().parents[1];PY='/mnt/ojc/miniconda3/envs/vllm_vlm/bin/python';status=W/'reports/background_status.json'
def run(label,args):
 print(time.strftime('%FT%TZ',time.gmtime()),'START',label,flush=True)
 with (W/'logs'/('background_'+label+'.log')).open('w') as f:
  p=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 print(time.strftime('%FT%TZ',time.gmtime()),'END',label,p.returncode,flush=True)
 return dict(task=label,returncode=p.returncode)
def render_branch():
 results=[]
 results.append(run('unu_H_render',[PY,str(W/'src/render_ready.py'),str(W/'runs/unu'),'2','H0']))
 for name in ['gogoro','unu']:
  r=run(name+'_B0',[PY,str(W/'src/run_stage.py'),str(W/'runs'/name/'B0-match'),'B0','--gpu','2']);results.append(r)
  if r['returncode']==0:results.append(run(name+'_B0_render',[PY,str(W/'src/render_ready.py'),str(W/'runs'/name),'2','B0-match']))
 return results
def retry_branch():
 return [run('govecs_flux_retry',[PY,str(W/'src/run_stage.py'),str(W/'runs/govecs/F2'),'flux','--gpu','1'])]
status.write_text(json.dumps(dict(state='running',started_utc=time.strftime('%FT%TZ',time.gmtime()),publication=False,review_required=True),indent=2))
with concurrent.futures.ThreadPoolExecutor(2) as pool:
 futures=[pool.submit(render_branch),pool.submit(retry_branch)];results=[r for f in futures for r in f.result()]
status.write_text(json.dumps(dict(state='awaiting_manual_review',results=results,publication=False,review_required=True,finished_utc=time.strftime('%FT%TZ',time.gmtime())),indent=2))
