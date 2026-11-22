import asyncio,json,os,time,subprocess
from pathlib import Path
D=Path(__file__).resolve().parents[1]
def save(x):
 p=D/'STATUS.json';q=p.with_suffix('.writing');q.write_text(json.dumps(x,ensure_ascii=False,indent=2));q.replace(p)
async def main():
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True);gpus=[(int(a),b.strip(),int(c)) for a,b,c in (x.split(',') for x in raw.strip().splitlines())];assert len(gpus)==8
 jobs=[f'{n}_{r}' for n in ['N06','N11'] for r in ['R1','R2','R3']];states={};save(dict(state='readonly_diagnostics_running',training_started=False,optimizer_calls=0,jobs=states))
 async def job(key,gpu):
  idx,uuid,free=gpu;assert free>8000
  if (D/'results'/key/'complete.json').exists():states[key]=dict(state='already_complete');return
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=uuid
  with (D/'logs'/f'{key}.log').open('a') as log:
   proc=await asyncio.create_subprocess_exec('/mnt/ojc/miniconda3/envs/grid4d/bin/python',str(D/'scripts/diagnostic.py'),'--key',key,stdout=log,stderr=log,env=env)
   states[key]=dict(state='running',gpu=idx,uuid=uuid,pid=proc.pid,started=time.time());save(dict(state='readonly_diagnostics_running',training_started=False,optimizer_calls=0,jobs=states))
   rc=await proc.wait();states[key].update(state='complete' if rc==0 else 'failed',exit_code=rc,ended=time.time());save(dict(state='readonly_diagnostics_running',training_started=False,optimizer_calls=0,jobs=states))
 await asyncio.gather(*(job(k,g) for k,g in zip(jobs,gpus)))
 save(dict(state='renders_complete_analysis_pending' if all(x['state'] in ['complete','already_complete'] for x in states.values()) else 'render_failure',training_started=False,optimizer_calls=0,jobs=states))
asyncio.run(main())
