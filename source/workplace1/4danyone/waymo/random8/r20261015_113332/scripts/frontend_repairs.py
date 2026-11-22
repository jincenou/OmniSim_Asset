"""Bounded recovery of the SAME frozen inputs after verified adapter fixes.

Only single-card frontend work; it cannot regenerate N04, start any backend,
or claim main-controller task success. It uses independent atomic receipts.
"""
import asyncio,fcntl,json,os,shutil,sys,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[1];W=R.parents[1];P=W.parent;S=R/'scripts';LOG=R/'logs/frontend_v12';LOG.mkdir(exist_ok=True)
sys.path.insert(0,str(W/'round2/scripts'));from resource_lease import lease,wait_memory
ENV=Path('/mnt/ojc/miniconda3/envs');PY={k:str(ENV/name/'bin/python') for k,name in [('front','4danyone'),('sam','4danyone-sam2')]};gpu_sem=asyncio.Semaphore(1);cpu_sem=asyncio.Semaphore(2);disk_sem=asyncio.Semaphore(1);STATE=R/'FRONTEND_REPAIR_STATUS.json';state={'state':'starting','pid':os.getpid(),'protocol':'random8-frontend-v1.2','cases':{},'no_model_regeneration':True,'selection_unchanged':True}
if STATE.exists():state.update(json.loads(STATE.read_text()));state.update(state='resuming',pid=os.getpid())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,x):
 t=p.with_suffix(p.suffix+'.writing');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def save():state['updated_at']=now();write(STATE,state)
def env(label):
 c=json.loads((R/'samples'/label/'candidate.json').read_text());source=R/'samples'/label/'source';gen=R/'samples'/label/'generation';frames=c['output_slots'];return dict(os.environ,CUDA_VISIBLE_DEVICES='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MPLCONFIGDIR=str(R/'cache/matplotlib'),TORCH_HOME=str(R/'cache/torch'),FDANYONE_SAMPLE_ID=label,FDANYONE_SAMPLE_ROOT=str(source),FDANYONE_MOVING_ROOT=str(source),FDANYONE_WAYMO_ROOT=str(gen),FDANYONE_SOURCE_EXPERIMENT=str(source),FDANYONE_RUNTIME=str(W/'round2/moving61_runtime' if frames==61 else P/'repo'),FDANYONE_FRAMES=str(frames),FDANYONE_MOTION_VARIANT='calibrated_moving',FDANYONE_ROOT_VIEWS='24')
async def room(label):
 while shutil.disk_usage(R).free<36.6*1024**3:
  state['cases'][label].update(state='blocked_disk_frontend_repair',free_bytes=shutil.disk_usage(R).free);save();await asyncio.sleep(120)
async def stage(label,name,kind,script,args,proof,gpu=False,minimum=6000):
 receipt=LOG/f'{label}_{name}.json'
 if proof.exists():
  if not receipt.exists():write(receipt,{'state':'existing_verified_prefix_same_run','proof':str(proof),'exit_code':None,'note':'Not inventing historical subprocess exitcode'})
  return
 if receipt.exists() and json.loads(receipt.read_text()).get('state')=='failed':raise RuntimeError(f'{label}/{name} alreadyfailed,no automaticrestart')
 await room(label)
 async def run(handles=()):
  command=[PY[kind],'-u',str(S/script),*args];log=LOG/f'{label}_{name}.log'
  with log.open('xb') as stream:
   child=await asyncio.create_subprocess_exec(*command,env=env(label),cwd=R,stdout=stream,stderr=asyncio.subprocess.STDOUT,pass_fds=handles,start_new_session=True);job={'state':'running','pid':child.pid,'process_group':child.pid,'physical_gpus':[4] if gpu else [],'command':command,'log':str(log),'started_at':now()};state['cases'][label].update(state='running',stage=name,job=job);write(receipt,job);save();code=await child.wait()
  job.update(state='complete' if code==0 else 'failed',exit_code=code,ended_at=now());write(receipt,job);save()
  if code or not proof.exists():raise RuntimeError(f'{label}/{name} exit{code},proof={proof.exists()};{log}')
 if gpu:
  async with gpu_sem:
   async with lease([4]) as handles:await wait_memory([4],minimum);await run(handles)
 else:
  async with cpu_sem:await run()
def archive(path,label):
 if not path.exists():return
 destination=path.with_name(path.name+'_failed_before_v12');assert not destination.exists();path.rename(destination);state['cases'][label].setdefault('archived',[]).append({'old':str(path),'retained':str(destination)});save()
async def person(label):
 base=R/'samples'/label;source=base/'source';generation=base/'generation';state['cases'].setdefault(label,{});state['cases'][label].update(state='queued');save()
 try:
  assert label!='N04';assert not (generation/'outputs/turbo/metadata.json').exists(),'Never regenerate an already-generated person'
  if (source.exists() and not (source/'native.json').exists()):archive(source,label)
  if (source/'inputs').exists() and not (source/'inputs/frame_mapping.json').exists() and not (source/'inputs_rife/frame_mapping.json').exists():archive(source/'inputs',label)
  if generation.exists() and not (generation/'preflight/review.json').exists():archive(generation,label)
  await stage(label,'native','front','m02_prepare.py',[],source/'native.json')
  await stage(label,'interpolate','front','interpolate.py',[],source/('inputs_rife/frame_mapping.json' if (source/'inputs_rife').exists() else 'inputs/frame_mapping.json'),True)
  await stage(label,'prepare','front','source_foreground.py',['prepare'],source/'segmentation/frames/00000.jpg')
  await stage(label,'segment','sam','source_foreground.py',['segment'],source/'segmentation/target_masks.npz',True,7000)
  await stage(label,'composite','front','source_foreground.py',['composite'],source/'research/batch_gate_mask.json')
  gate=json.loads((source/'research/batch_gate_mask.json').read_text());assert gate['passed'],'Existing foreground gate failed;not a repaired numeric gate'
  await stage(label,'motion','front','motion.py',[],source/'motion_pair/calibrated_moving/motion.safetensors',True,15000)
  await stage(label,'motion_gate','front','source_foreground.py',['motion_gate'],source/'research/motion_gate.json')
  await stage(label,'conditioning','front','conditioning.py',[],generation/'preflight/review.json',True,9000)
  assert json.loads((generation/'preflight/review.json').read_text())['allow_generation'];state['cases'][label].update(state='same_input_repaired_ready_for_first24view_generation',ready_at=now());save()
 except Exception as e:state['cases'][label].update(state='failed_after_bounded_repair',error=f'{type(e).__name__}:{e}');save()
async def main():
 own=(R/'frontend_repairs.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);state['state']='repairing_same_preregistered_inputs';save()
 # N09 is the first preregistered reserve, justified by N05's unchanged hard foreground failure.
 await asyncio.gather(*map(person,['N01','N02','N03','N06','N07','N08','N09']))
 state['state']='bounded_frontend_repairs_terminal';save()
if __name__=='__main__':asyncio.run(main())
