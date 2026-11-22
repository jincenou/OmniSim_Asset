"""Persistent eight-case async motion-only experiment; no generation approval."""
import asyncio,fcntl,hashlib,json,os,shutil,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
W=ROOT.parent;RUN=W/'scene20/r20261019';SCRIPTS=RUN/'scripts'
PY=Path('/mnt/ojc/miniconda3/envs/4danyone/bin/python')
CASES=['005/P18','178/P22','088/P41','038/P08','061/P36','176/P04','155/P30','170/P24']
STATE={'scope':'motion_input_ablation_only','state':'running','jobs':{}}
CPU=asyncio.Semaphore(2)

def save(path,data):
 tmp=path.with_suffix('.writing');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def publish():
 STATE['updated_at_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save(ROOT/'STATUS.json',STATE)

async def run(case,gpu):
 baseline=RUN/'scenes'/case;name=baseline.name
 target=ROOT/'motion_unmasked'/name
 if target.exists():raise FileExistsError(f'Existing experiment must be inspected before recovery: {target}')
 target.parent.mkdir(exist_ok=True)
 frames=json.loads((baseline/'generation.json').read_text())['output']['frames_per_video']
 env=os.environ.copy();env.pop('PYTHONPATH',None);env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
 env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
  HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',FDANYONE_SAMPLE_ID=name,FDANYONE_SAMPLE_ROOT=str(target),FDANYONE_MOVING_ROOT=str(target),
  FDANYONE_BASELINE=str(baseline),FDANYONE_FRAMES=str(frames),FDANYONE_RUNTIME=str(W/'round2/moving61_runtime' if frames==61 else W.parent/'repo'),
  TORCH_EXTENSIONS_DIR=str(RUN/'cache/torch_extensions'),TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='4',TMPDIR=str(ROOT/'tmp'))
 lock=(W/'gpu_leases_20261019'/f'gpu{gpu}.lock').open('a')
 try:
  while True:
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:
    STATE['jobs'][name]={'state':'waiting_gpu_lease','gpu':gpu};publish();await asyncio.sleep(20);continue
   p=await asyncio.create_subprocess_exec('nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE)
   out,_=await p.communicate();assert p.returncode==0
   if int(out.strip())>=24000:break
   fcntl.flock(lock,fcntl.LOCK_UN);STATE['jobs'][name]={'state':'waiting_memory','gpu':gpu};publish();await asyncio.sleep(20)
  stages=[('crop',SCRIPTS/'m02_prepare.py',True),('interpolate',SCRIPTS/'interpolate.py',False),
          ('verify',ROOT/'verify_input.py',True),('motion',ROOT/'motion_unmasked_entry.py',False)]
  for stage,script,cpu in stages:
   if shutil.disk_usage(ROOT).free<5*2**30:raise RuntimeError('Less than 5 GiB free for safe writes')
   async def execute():
    with (ROOT/'logs'/f'{name}_{stage}.log').open('x') as log:
     begin=time.time();p=await asyncio.create_subprocess_exec(str(PY),str(script),env=env,stdout=log,stderr=log)
     STATE['jobs'][name]={'state':'running','gpu':gpu,'stage':stage,'pid':p.pid,'started':begin};publish()
     rc=await p.wait();save(ROOT/'receipts'/f'{name}_{stage}.json',dict(exit_code=rc,pid=p.pid,gpu=gpu,seconds=time.time()-begin,script_sha256=hashlib.sha256(script.read_bytes()).hexdigest()))
     if rc:raise RuntimeError(f'{stage} exit {rc}')
   if cpu:
    async with CPU:await execute()
   else:await execute()
  STATE['jobs'][name].update(state='motion_complete_needs_comparison',finished=time.time());publish()
 except Exception as error:
  STATE['jobs'][name]={'state':'failed','gpu':gpu,'error':repr(error)};publish()
 finally:
  fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

async def main():
 for directory in ['logs','receipts','tmp']: (ROOT/directory).mkdir(exist_ok=True)
 if (ROOT/'STATUS.json').exists():raise FileExistsError('Inspect prior run; do not overwrite status')
 publish()
 await asyncio.gather(*(run(case,gpu) for gpu,case in enumerate(CASES)))
 STATE['state']='motion_queue_finished_with_failures' if any(j['state']=='failed' for j in STATE['jobs'].values()) else 'motion_queue_finished_needs_review';publish()
if __name__=='__main__':asyncio.run(main())
