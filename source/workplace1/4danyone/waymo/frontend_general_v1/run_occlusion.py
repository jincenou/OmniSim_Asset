"""Bounded async GPU counterfactual jobs, independent tmux, fail closed."""
from pathlib import Path
import asyncio,json,time,os,hashlib,shutil,fcntl
G=Path(__file__).resolve().parent;O=G/'occlusion_v1';W=G.parent;D=W/'frontend_quality_20261020';S=W/'scene20/r20261019/scripts'
PY='/mnt/ojc/miniconda3/envs/4danyone/bin/python'
STATE={'state':'running','jobs':{}};CPU=asyncio.Semaphore(2)
def save(p,v):
 t=p.with_suffix('.writing');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def update():STATE['updated_at']=time.time();save(O/'STATUS.json',STATE)
async def execute(job,gpu):
 key=job['key'];root=Path(job['root']);sample=job['sample'];branch=job['branch']
 env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
 env.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',FDANYONE_SAMPLE_ROOT=str(root),FDANYONE_MOVING_ROOT=str(root),FDANYONE_SAMPLE_ID=sample,FDANYONE_RUNTIME=str(W/'round2/moving61_runtime'),FDANYONE_FRAMES='61',
 TORCH_EXTENSIONS_DIR=str(W/'scene20/r20261019/cache/torch_extensions'),TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='2',TMPDIR=str(G/'tmp'))
 async def run(stage,command):
  if shutil.disk_usage(G).free<5*2**30:raise RuntimeError('Less than5GiB safe writes')
  with (root/f'{stage}.log').open('x') as log:
   p=await asyncio.create_subprocess_exec(PY,*map(str,command),env=env,stdout=log,stderr=asyncio.subprocess.STDOUT)
   STATE['jobs'][key]={'state':'running','stage':stage,'gpu':gpu,'pid':p.pid,'started':time.time()};update()
   code=await p.wait();save(root/f'{stage}_receipt.json',{'exit_code':code,'pid':p.pid,'ended':time.time()})
   if code:raise RuntimeError(f'{stage} exited{code}')
 lock=(W/'gpu_leases_20261019'/f'gpu{gpu}.lock').open('a')
 try:
  while True:
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:await asyncio.sleep(15);continue
   p=await asyncio.create_subprocess_exec('nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE)
   data,_=await p.communicate()
   if p.returncode:raise RuntimeError('GPU query failed')
   if int(data.strip())>=16000:break
   fcntl.flock(lock,fcntl.LOCK_UN);STATE['jobs'][key]={'state':'waiting_memory','gpu':gpu};update();await asyncio.sleep(15)
  if branch=='rgb':await run('interpolate',[S/'interpolate.py'])
  await run('motion',[G/'occlusion_motion.py'])
  fcntl.flock(lock,fcntl.LOCK_UN)
  # Isolated adapter paths: optimizer never reads hidden reference image/features.
  (root/'motion').symlink_to(root/'motion_pair/calibrated_moving',target_is_directory=True)
  inp=root/'input';inp.mkdir();(inp/'model_input.mp4').symlink_to(root/'inputs/main.mp4');(inp/'frame_mapping.json').symlink_to(root/'inputs/frame_mapping.json');(inp/'target_masks.npz').symlink_to(root.parent/'evidence_masks.npz')
  save(root/'fit_config.json',{'sample':sample,'branch':branch,'protocol':str(O/'PROTOCOL.json'),'visibility':'known target bbox excluding known synthetic occluder, not hidden original instance mask'})
  async with CPU:await run('optimize',[G/'optimize_motion.py',sample,'--variant','local_temporal','--case-config',root/'fit_config.json'])
  STATE['jobs'][key].update(state='complete_needs_heldout_evaluation',finished=time.time());update()
 except Exception as e:STATE['jobs'][key]={'state':'failed','error':repr(e),'gpu':gpu};update()
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
async def worker(gpu,queue):
 while not queue.empty():
  try:job=queue.get_nowait()
  except asyncio.QueueEmpty:return
  await execute(job,gpu);queue.task_done()
async def main():
 if (O/'STATUS.json').exists():raise FileExistsError('Existing queue requires inspection')
 files=[G/'occlusion_motion.py',G/'prepare_occlusion.py',G/'optimize_motion.py',G/'run_occlusion.py',G/'gap_proposal.py',G/'camera_geometry.py',G/'protocol_v2.json',O/'PROTOCOL.json',S/'interpolate.py']
 save(O/'implementation_freeze.json',{'before_execution_utc_seconds':time.time(),'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})
 queue=asyncio.Queue()
 for job in json.loads((O/'jobs.json').read_text()):queue.put_nowait(job)
 await asyncio.gather(*(worker(gpu,queue) for gpu in [1,2,3,4]))
 STATE['state']='finished_with_failures' if any(x['state']=='failed' for x in STATE['jobs'].values()) else 'finished_needs_evaluation';update()
if __name__=='__main__':asyncio.run(main())
