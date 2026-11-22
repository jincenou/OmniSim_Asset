"""Eight fixed Base-v-Turbo tests, async per GPU, no baseline mutation."""
import asyncio,fcntl,hashlib,json,os,time,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;W=ROOT.parent;RUN=W/'scene20/r20261019';PY='/mnt/ojc/miniconda3/envs/4danyone/bin/python'
CASES=['005/P18','178/P22','088/P41','038/P08','061/P36','176/P04','155/P30','170/P24']
STATE={'state':'running','jobs':{},'scope':'fixed_baseline_motion_and_input; Base24 versus Turbo4'}
def dump(p,x):
 t=p.with_suffix('.writing');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def save():STATE['updated_at']=time.time();dump(ROOT/'base_status.json',STATE)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
async def run(case,gpu):
 old=RUN/'scenes'/case;name=old.name;work=ROOT/'base_ablation'/name;source=work/'source';gen=work/'generation'
 lease=(W/'gpu_leases_20261019'/f'gpu{gpu}.lock').open('a')
 try:
  if work.exists():raise FileExistsError(work)
  while True:
   try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:await asyncio.sleep(20);continue
   p=await asyncio.create_subprocess_exec('nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE)
   stdout,_=await p.communicate();assert p.returncode==0
   if int(stdout.strip())>=35000 and shutil.disk_usage(ROOT).free>=8*2**30:break
   fcntl.flock(lease,fcntl.LOCK_UN);STATE['jobs'][name]={'state':'waiting_resources','gpu':gpu};save();await asyncio.sleep(20)
  (source/'inputs').mkdir(parents=True);(source/'research').mkdir();(source/'motion_pair').mkdir()
  for dest,origin in [('main.mp4','model_input.mp4'),('frame_mapping.json','frame_mapping.json')]: (source/'inputs'/dest).symlink_to(old/'input'/origin)
  (source/'candidate.json').symlink_to(old/'candidate.json');(source/'motion_pair/calibrated_moving').symlink_to(old/'motion',target_is_directory=True)
  dump(source/'research/motion_gate.json',{'input_sha256':sha(old/'input/model_input.mp4'),'variants':{'calibrated_moving':{'allow_exploratory_generation':True,'motion_sha256':sha(old/'motion/motion.safetensors')}},'purpose':'Paired Base ablation of known imperfect baseline; not motion-quality approval'})
  masks=np.load(old/'input/target_masks.npz')['masks']>127
  mapping=json.loads((old/'input/frame_mapping.json').read_text())
  boxes=np.asarray([x['bbox_input_xyxy'] for x in mapping['frames']])
  area=masks.sum((1,2))/((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]));core=[]
  for mask,b in zip(masks,boxes,strict=True):
   x0,y0,x1,y1=b;w=x1-x0;h=y1-y0;xa,ya,xb,yb=np.rint([x0+.3*w,y0+.2*h,x0+.7*w,y0+.6*h]).astype(int)
   patch=mask[max(ya,0):min(yb,mask.shape[0]),max(xa,0):min(xb,mask.shape[1])];core.append(float(patch.mean()) if patch.size else 0.)
  ok=bool(area.min()>.15 and np.median(area)>.3 and area.min()/np.median(area)>.55 and min(core)>.3 and np.quantile(core,.1)>.55)
  dump(source/'research/batch_gate_mask.json',{'passed':ok,'mask_area_min':float(area.min()),'mask_area_median':float(np.median(area)),'torso_min':min(core),'torso_p10':float(np.quantile(core,.1)),'scope':'Original numeric gate recomputed on retained masks; not motion or appearance quality approval'})
  if not ok:raise RuntimeError('Retained baseline does not reproduce original mask gate')
  count=json.loads((old/'generation.json').read_text())['output']['frames_per_video']
  env=os.environ.copy();env.pop('PYTHONPATH',None);env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
  env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
   FDANYONE_BASELINE=str(old),FDANYONE_WAYMO_ROOT=str(gen),FDANYONE_SOURCE_EXPERIMENT=str(source),FDANYONE_SAMPLE_ROOT=str(source),FDANYONE_FRAMES=str(count),
   FDANYONE_RUNTIME=str(W/'round2/moving61_runtime' if count==61 else W.parent/'repo'),FDANYONE_MOTION_VARIANT='calibrated_moving',FDANYONE_ROOT_VIEWS='24',FDANYONE_RETAINED_VIEWS='24',
   TORCH_EXTENSIONS_DIR=str(RUN/'cache/torch_extensions'),TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='4',TMPDIR=str(ROOT/'tmp'))
  stages=[('conditioning',RUN/'scripts/conditioning.py',[]),('pairing',ROOT/'check_conditioning.py',[]),('generate',ROOT/'generate_base.py',['base'])]
  for stage,script,args in stages:
   with (ROOT/'logs'/f'{name}_base_{stage}.log').open('x') as log:
    begin=time.time();p=await asyncio.create_subprocess_exec(PY,str(script),*args,env=env,stdout=log,stderr=log)
    STATE['jobs'][name]={'state':'running','stage':stage,'gpu':gpu,'pid':p.pid,'started':begin};save();rc=await p.wait()
    dump(ROOT/'receipts'/f'{name}_base_{stage}.json',dict(exit_code=rc,seconds=time.time()-begin,pid=p.pid,gpu=gpu,script_sha256=sha(script)))
    if rc:raise RuntimeError(f'{stage} exit {rc}')
  STATE['jobs'][name].update(state='generated_needs_validation',finished=time.time());save()
 except Exception as error:STATE['jobs'][name]={'state':'failed','gpu':gpu,'error':repr(error)};save()
 finally:fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
async def main():
 if (ROOT/'base_status.json').exists():raise FileExistsError('Inspect prior run before recovery')
 save();await asyncio.gather(*(run(case,gpu) for gpu,case in enumerate(CASES)));STATE['state']='queue_finished_needs_review';save()
if __name__=='__main__':asyncio.run(main())
