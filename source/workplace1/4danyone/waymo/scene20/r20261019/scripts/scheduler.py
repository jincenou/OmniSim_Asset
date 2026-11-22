"""Async per-device workers, shared leases, persistent stage receipts."""
import asyncio,fcntl,json,os,time,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];W=R.parents[1];P=W.parent
ENV=Path('/mnt/ojc/miniconda3/envs');LOCKS=W/'gpu_leases_20261019'
STATUS=R/os.environ.get('FD_SCENE20_STATUS','STATUS.json')
state={'state':'running','jobs':{},'completed':0,'total':20};cpu=None

def save(p,x):
 t=p.with_suffix('.writing');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));t.replace(p)
def status():
 state['updated_at']=time.time();state['completed']=sum(x.get('state')=='complete' for x in state['jobs'].values());save(STATUS,state)
def env_for(row,gpu):
 b=R/'scenes'/row['scene']/row['sample_id'];src=b/'.work/source';gen=b/'.work/generation';runtime=W/'round2/moving61_runtime' if row['output_slots']==61 else P/'repo';e=os.environ.copy()
 e.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',FDANYONE_SAMPLE_ID=row['sample_id'],FDANYONE_SAMPLE_ROOT=str(src),FDANYONE_MOVING_ROOT=str(src),FDANYONE_FRAMES=str(row['output_slots']),FDANYONE_SOURCE_EXPERIMENT=str(src),FDANYONE_WAYMO_ROOT=str(gen),FDANYONE_RUNTIME=str(runtime),FDANYONE_MOTION_VARIANT='calibrated_moving',FDANYONE_ROOT_VIEWS='24',FDANYONE_RETAINED_VIEWS='24',TORCH_EXTENSIONS_DIR=str(R/'cache/torch_extensions'),TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='4',TMPDIR=str(R/'tmp'))
 e.pop('PYTORCH_CUDA_ALLOC_CONF',None);e.pop('PYTHONPATH',None);return e
async def gpu_free(g):
 p=await asyncio.create_subprocess_exec('nvidia-smi',f'--id={g}','--query-gpu=memory.free','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE);out,_=await p.communicate();assert p.returncode==0;return int(out.strip())
async def run(row,g):
 key=row['sample_id'];b=R/'scenes'/row['scene']/key;b.mkdir(parents=True,exist_ok=True);save(b/'candidate.json',row);work=b/'.work';work.mkdir(exist_ok=True);shutil.copy2(b/'candidate.json',work/'candidate.json');e=env_for(row,g);receipts=R/'research'/key;receipts.mkdir(exist_ok=True)
 stages=[('camera','4dgaussians',['raw_camera_audit.py','--review-id',key,'--manifest',str(R/('reserve_manifest.json' if row['cohort']=='scene20_reserve' else 'selection_manifest.json'))],True),('crop','4danyone',['m02_prepare.py'],True),('interpolate','4danyone',['interpolate.py'],False),('seg_prepare','4danyone',['source_foreground.py','prepare'],True),('segment','4danyone-sam2',['source_foreground.py','segment'],False),('composite','4danyone',['source_foreground.py','composite'],True),('motion','4danyone',['motion.py'],False),('motion_gate','4danyone',['source_foreground.py','motion_gate'],True),('conditioning','4danyone',['conditioning.py'],False),('generate','4danyone',['generate_single.py','turbo'],False),('deliver','4danyone',['deliver.py','--sample',str(b),'--frames',str(row['output_slots'])],True)]
 for name,envname,args,is_cpu in stages:
  marker=receipts/f'{name}.json'
  if marker.exists() and json.loads(marker.read_text())['exit_code']==0:continue
  while shutil.disk_usage(R).free<8*2**30:
   state['jobs'][key]={'state':'waiting_disk','stage':name,'gpu':g,'minimum_free_gib':8};status();await asyncio.sleep(30)
  # Eight GiB admission headroom is temporary-work allowance, not the canceled35GiB rule.
  async def execute():
   with (R/'logs'/f'{key}_{name}.log').open('a') as log:
    started=time.time();p=await asyncio.create_subprocess_exec(str(ENV/envname/'bin/python'),str(R/'scripts'/args[0]),*args[1:],env=e,stdout=log,stderr=log)
    state['jobs'][key]={'state':'running','stage':name,'gpu':g,'pid':p.pid,'scene':row['scene'],'started':started};status();rc=await p.wait();save(marker,{'exit_code':rc,'stage':name,'seconds':time.time()-started,'pid':p.pid,'gpu':g});return rc
  if is_cpu:
   async with cpu:rc=await execute()
  else:rc=await execute()
  if rc:state['jobs'][key].update(state='failed',exit_code=rc);status();return
 state['jobs'][key].update(state='complete',finished=time.time());status()
async def main():
 global cpu
 cpu=asyncio.Semaphore(2);rows=json.loads((R/'selection_manifest.json').read_text())['candidates'];q=asyncio.Queue()
 batch=os.environ.get('FD_SCENE20_BATCH_IDS')
 if batch:
  rows+=json.loads((R/'reserve_manifest.json').read_text())['candidates'];chosen=set(batch.split(','));rows=[r for r in rows if r['sample_id'] in chosen]
  assert len(rows)==len(chosen)
 previous=json.loads(STATUS.read_text()) if STATUS.exists() else {'jobs':{}}
 state['jobs'].update(previous['jobs'])
 for row in rows:
  if state['jobs'].get(row['sample_id'],{}).get('state') not in ('complete','failed'):q.put_nowait(row)
 async def worker(g):
  lease=(LOCKS/f'gpu{g}.lock').open('a')
  try:
   while not q.empty():
    try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:await asyncio.sleep(15);continue
    try:
     if await gpu_free(g)<35000:await asyncio.sleep(15);continue
     try:row=q.get_nowait()
     except asyncio.QueueEmpty:return
     try:await run(row,g)
     except Exception as ex:state['jobs'][row['sample_id']]={'state':'failed','error':repr(ex),'gpu':g};status()
     finally:q.task_done()
    finally:fcntl.flock(lease,fcntl.LOCK_UN)
  finally:lease.close()
 await asyncio.gather(*(worker(g) for g in [6,7,0,1,2,3,4,5]));state['state']='queue_finished_with_failures' if any(j.get('state')=='failed' for j in state['jobs'].values()) else 'queue_finished';status()
if __name__=='__main__':asyncio.run(main())
