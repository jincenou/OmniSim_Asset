import asyncio,fcntl,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CASES=['005/P18','178/P22','088/P41','038/P08','061/P36','176/P04','155/P30','170/P24']
STATE={'state':'running','jobs':{}}
def save():
 p=ROOT/'appearance_status.json';t=p.with_suffix('.writing');t.write_text(json.dumps(STATE,indent=2)+'\n');t.replace(p)
async def run(case,gpu):
 key=case.split('/')[-1];lease=(ROOT.parent/'gpu_leases_20261019'/f'gpu{gpu}.lock').open('a')
 try:
  while True:
   try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:await asyncio.sleep(20);continue
   p=await asyncio.create_subprocess_exec('nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE)
   out,_=await p.communicate();assert p.returncode==0
   if int(out.strip())>24000:break
   fcntl.flock(lease,fcntl.LOCK_UN);STATE['jobs'][key]={'state':'waiting_gpu_memory','gpu':gpu};save();await asyncio.sleep(20)
  env=os.environ.copy();env.pop('PYTHONPATH',None)
  env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',FDANYONE_SAMPLE_ROOT=str(ROOT/'motion_unmasked'/key))
  with (ROOT/'logs'/f'{key}_appearance.log').open('x') as log:
   begin=time.time();p=await asyncio.create_subprocess_exec('/mnt/ojc/miniconda3/envs/4danyone/bin/python',str(ROOT/'appearance_vae_probe.py'),case,env=env,stdout=log,stderr=log)
   STATE['jobs'][key]={'state':'running','pid':p.pid,'gpu':gpu,'started':begin};save();rc=await p.wait()
   STATE['jobs'][key].update(state='complete' if rc==0 else 'failed',exit_code=rc,seconds=time.time()-begin);save()
 except Exception as e:STATE['jobs'][key]={'state':'failed','error':repr(e)};save()
 finally:fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
async def main():
 if (ROOT/'appearance_status.json').exists():raise FileExistsError('Inspect existing appearance run before recovery')
 save();await asyncio.gather(*(run(case,gpu) for gpu,case in enumerate(CASES)));STATE['state']='finished';save()
if __name__=='__main__':asyncio.run(main())
