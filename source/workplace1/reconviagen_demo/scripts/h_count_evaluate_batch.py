"""Apply visually reviewed headings, then evaluate all new H-count assets."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,os,subprocess,time
from h_count_ablation import ROOT,WORK,PY,RENDER_PY,write

decisions=json.loads((WORK/'heading_decisions.json').read_text())['decisions']
cases=list(dict.fromkeys(r['case_id'] for r in decisions))

def worker(item):
 gpu,assigned=item
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
   PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PATH='/home/wmj/miniconda3/envs/sim/bin:'+env['PATH'],
   CPATH='/home/wmj/miniconda3/envs/splatad/include',TORCH_EXTENSIONS_DIR=str(ROOT/'cache/torch_extensions'),MAX_JOBS='2')
 for cid in assigned:
  log=WORK/'logs'/f'{cid}_evaluate.log';status=WORK/'status'/f'{cid}_evaluate.json'
  print('EVAL START',gpu,cid,flush=True)
  write(status,dict(status='running',gpu=gpu,started_unix=time.time()))
  try:
   with log.open('a') as f:
    def run(cmd):subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    for d in [d for d in decisions if d['case_id']==cid]:
     v=d['variant'];out=ROOT/'results'/cid/f'{v}_seed42'
     if not (out/'asset/heading_review.json').exists():
      run([str(PY),str(ROOT/'scripts/finalize_heading.py'),'--case',cid,'--variant',v,'--note',d['review']+(' '+d['defect'] if d['defect'] else '')]+(['--flip'] if d['flip'] else []))
     if not (out/'evaluation/render_audit.json').exists():
      run([RENDER_PY,str(ROOT/'scripts/render_metric_views.py'),'--case',cid,'--variant',v])
    for encoder in ['clip','dino']:
     variants=[f'H_real{n}' for n in [1,2,3] if not (ROOT/'results'/cid/f'H_real{n}_seed42/evaluation'/f'{encoder}_metrics.json').exists()]
     if variants:run(['/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python',str(ROOT/'scripts/evaluate_asset_metrics.py'),'--case',cid,'--variants',*variants,'--encoder',encoder])
    run([str(ROOT/'env_refine/bin/python'),str(ROOT/'scripts/h_count_metrics.py'),'pixels','--case',cid])
   write(status,dict(status='complete',gpu=gpu,finished_unix=time.time()))
   print('EVAL DONE',gpu,cid,flush=True)
  except Exception as e:
   write(status,dict(status='failed',gpu=gpu,error=str(e),log=str(log)))
   print('EVAL FAILED',gpu,cid,str(e),flush=True)

with ThreadPoolExecutor(max_workers=8) as pool:
 list(pool.map(worker,[(g,cases[g::8]) for g in range(8)]))
