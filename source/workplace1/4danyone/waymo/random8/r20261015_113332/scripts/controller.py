"""Persistent, bounded asynchronous queue for the frozen random8 protocol."""
import asyncio,fcntl,hashlib,json,os,shutil,signal,sys,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from contextlib import asynccontextmanager
R=Path(__file__).resolve().parents[1];W=R.parents[1];P=W.parent;S=R/'scripts';OLD=W/'round2';ENV=Path('/mnt/ojc/miniconda3/envs')
sys.path.insert(0,str(OLD/'scripts'));from resource_lease import lease
PY={k:str(ENV/name/'bin/python') for k,name in [('front','4danyone'),('sam','4danyone-sam2'),('ftgs','4danyone-ftgs'),('hust','4dgaussians')]};RESERVE=35*1024**3
state={'state':'starting','controller_pid':os.getpid(),'jobs':{},'samples':{},'planned_tracks':8,'planned_terminal_versions':32,'attempt_limit':16,'visual_review':'pending','production_approved':False}
if (R/'STATUS.json').exists():
 old=json.loads((R/'STATUS.json').read_text());state.update(old);state.update(controller_pid=os.getpid(),state='recovering')
selection=json.loads((R/'selection.json').read_text());memory_lock=asyncio.Lock();disk_lock=asyncio.Lock();disk_allocations={};front_limit=asyncio.Semaphore(1);case_limit=asyncio.Semaphore(1);cpu_limit=asyncio.Semaphore(2)
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def write(p,v):
 t=p.with_suffix(p.suffix+'.writing');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def persist():state['updated_at']=now();state['disk_free_bytes']=shutil.disk_usage(R).free;write(R/'STATUS.json',state)
def env_for(label,gpus):
 c=json.loads((R/'samples'/label/'candidate.json').read_text());frames=c['output_slots'];source=R/'samples'/label/'source';gen=R/'samples'/label/'generation';runtime=OLD/'moving61_runtime' if frames==61 else P/'repo'
 e=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',FDANYONE_SAMPLE_ID=label,FDANYONE_SAMPLE_ROOT=str(source),FDANYONE_MOVING_ROOT=str(source),FDANYONE_FRAMES=str(frames),FDANYONE_SOURCE_EXPERIMENT=str(source),FDANYONE_WAYMO_ROOT=str(gen),FDANYONE_RUNTIME=str(runtime),FDANYONE_MOTION_VARIANT='calibrated_moving',FDANYONE_ROOT_VIEWS='24',FDANYONE_RETAINED_VIEWS='24',FDANYONE_ASSET_NAME=label,FDANYONE_ASSET_SOURCE=str(gen/'outputs/turbo'),FDANYONE_DATA=str(R/'samples'/label/'data'),FDANYONE_ASSET_GPU_IDS=','.join(map(str,gpus)),TORCH_EXTENSIONS_DIR=str(R/'cache/torch_extensions'),TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='4')
 e.pop('PYTORCH_CUDA_ALLOC_CONF',None);return e
class StageError(RuntimeError):pass
@asynccontextmanager
async def disk_budget(key,gib):
 amount=int(gib*1024**3)
 while True:
  async with disk_lock:
   free=shutil.disk_usage(R).free
   if free-RESERVE-sum(disk_allocations.values())>=amount:
    disk_allocations[key]=amount;break
   state['jobs'].setdefault(key,{}).update(state='blocked_disk_admission',estimated_additional_bytes=amount,other_reserved_bytes=sum(disk_allocations.values()),free_bytes=free,reserve_bytes=RESERVE,updated_at=now());persist()
  await asyncio.sleep(120)
 try:yield
 finally:disk_allocations.pop(key,None)
async def free_memory(ids):
 p=await asyncio.create_subprocess_exec('nvidia-smi','--id='+','.join(map(str,ids)),'--query-gpu=memory.free','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE);out,_=await p.communicate()
 if p.returncode:raise RuntimeError('GPU query failed')
 return min(map(int,out.splitlines()))
@asynccontextmanager
async def devices(key,ids,minimum):
 while True:
  async with lease(ids) as handles:
   if await free_memory(ids)>=minimum:
    yield handles;return
   state['jobs'].setdefault(key,{}).update(state='waiting_gpu_memory',physical_gpus=ids,minimum_free_mib=minimum,updated_at=now());persist()
  await asyncio.sleep(120)
async def stage(label,name,kind,script,args=(),gpus=(),minimum=0,proof=None,environment=None):
 key=f'{label}/{name}';receipt=R/'logs'/f'{label}_{name}.json'
 if receipt.exists() and json.loads(receipt.read_text()).get('state')=='running':
  previous=json.loads(receipt.read_text());pid=previous['pid']
  while Path(f'/proc/{pid}/cmdline').exists() and Path(f'/proc/{pid}/cmdline').read_bytes():await asyncio.sleep(5)
  text=Path(previous['log']).read_text(errors='replace')
  if proof is not None and proof.exists() and 'Traceback (most recent call last)' not in text:
   previous.update(state='reconciled_finished_process_artifact_validated_exit_code_unavailable',exit_code=None,reconciled_at=now());write(receipt,previous);state['jobs'][key]=previous;persist();return
  raise StageError(f'{key} interrupted controller; child result needs review; log={previous["log"]}')
 if receipt.exists() and (json.loads(receipt.read_text()).get('exit_code')==0 or json.loads(receipt.read_text()).get('state','').startswith('reconciled_')):
  if proof is None or proof.exists():state['jobs'][key]=json.loads(receipt.read_text());persist();return
 if proof is not None and proof.exists() and name in ['camera_audit','native','interpolate']:
  # Only prefixes performed in this run, before the controller was created.
  state['jobs'][key]={'state':'verified_existing_prefix_this_run','proof':str(proof),'updated_at':now()};write(receipt,{**state['jobs'][key],'exit_code':0});persist();return
 async def run(handles=()):
  attempt=1
  while (R/'logs'/f'{label}_{name}_attempt{attempt}.log').exists():attempt+=1
  if receipt.exists():
   history=receipt.with_name(receipt.stem+f'_before_attempt{attempt}.json')
   if not history.exists():history.write_bytes(receipt.read_bytes())
  log=R/'logs'/f'{label}_{name}_attempt{attempt}.log';command=[PY[kind],'-u',str(S/script),*map(str,args)];env=env_for(label,gpus)
  if environment:env.update(environment)
  with log.open('xb') as out:
   child=await asyncio.create_subprocess_exec(*command,cwd=R,env=env,stdout=out,stderr=asyncio.subprocess.STDOUT,start_new_session=True,pass_fds=handles)
   record={'state':'running','command':command,'environment':{k:v for k,v in env.items() if k.startswith(('FDANYONE_','CUDA_','TORCH_'))},'pid':child.pid,'process_group':child.pid,'physical_gpus':list(gpus),'started_at':now(),'log':str(log),'attempt':attempt};state['jobs'][key]=record;write(receipt,record);persist();print(json.dumps({'job':key,**record}),flush=True)
   code=await child.wait()
  record.update(state='complete' if code==0 else 'failed',exit_code=code,ended_at=now());write(receipt,record);state['jobs'][key]=record;persist()
  if code:raise StageError(f'{key} exit{code}; {log}')
  if proof is not None and not proof.exists():raise StageError(f'{key} exited0 but missing proof {proof}')
 if gpus:
  async with devices(key,list(gpus),minimum) as handles:await run(handles)
 else:await run()
async def audit(c):
 label=c['sample_id']
 async with cpu_limit:
  try:await stage(label,'camera_audit','hust','raw_camera_audit.py',['--review-id',label,'--manifest',R/'selection_manifest.json'],proof=R/'camera_audit'/label/'audit.json')
  except Exception as error:state['samples'].setdefault(label,{}).update(camera_audit_error=str(error));persist()
async def frontend(c):
 label=c['sample_id'];base=R/'samples'/label;src=base/'source';gen=base/'generation';frames=c['output_slots']
 async with front_limit:
  await stage(label,'camera_audit','hust','raw_camera_audit.py',['--review-id',label,'--manifest',R/'selection_manifest.json'],proof=R/'camera_audit'/label/'audit.json')
  await stage(label,'native','front','m02_prepare.py',proof=src/'native.json')
  await stage(label,'interpolate','front','interpolate.py',gpus=[5],minimum=6000,proof=src/'inputs_rife/frame_mapping.json' if (src/'inputs_rife').exists() else src/'inputs/frame_mapping.json')
  await stage(label,'source_prepare','front','source_foreground.py',['prepare'],proof=src/'segmentation/frames/00000.jpg')
  await stage(label,'source_segment','sam','source_foreground.py',['segment'],gpus=[5],minimum=7000,proof=src/'segmentation/target_masks.npz')
  await stage(label,'source_composite','front','source_foreground.py',['composite'],proof=src/'research/batch_gate_mask.json')
  await stage(label,'motion','front','motion.py',gpus=[5],minimum=15000,proof=src/'motion_pair/calibrated_moving/motion.safetensors')
  await stage(label,'motion_gate','front','source_foreground.py',['motion_gate'],proof=src/'research/motion_gate.json')
  await stage(label,'conditioning','front','conditioning.py',gpus=[5],minimum=9000,proof=gen/'preflight/review.json')
  await stage(label,'generate24','front','generate.py',['turbo'],gpus=[4,5,6,7],minimum=32000,proof=gen/'outputs/turbo/metadata.json')
  from finish_generation import finish
  finish(base,frames)
  await stage(label,'rgba','sam','full_masks.py',gpus=[4,5,6,7],minimum=10000,proof=R/'research'/f'{label}-mask-complete.json')
async def branch(c,kind):
 label=c['sample_id'];base=R/'samples'/label
 try:
  isft=kind.startswith('ftgs');gpu={'ftgs_base5k':4,'ftgs_opt15k':5,'4dgs_grid25':7,'4dgs_grid61':6}[kind]
  async with disk_budget(f'{label}/{kind}',.40 if kind=='ftgs_base5k' else .65 if isft else 1.5):
   await stage(label,kind,'ftgs' if isft else 'hust','train_ftgs.py' if isft else 'train_4dgs.py',['--label',label,'--branch',kind] if isft else ['--label',label,'--grid',kind[-2:]],gpus=[gpu],minimum=6500 if isft else 12000,proof=base/kind/'training_complete.json' if isft else R/'four_dg/runs'/f'{label}_grid{kind[-2:]}'/'complete.json')
  async with disk_budget(f'{label}/{kind}_evaluation',.15):
   await stage(label,f'{kind}_evaluation','ftgs' if isft else 'hust','evaluate_branch.py',['--label',label,'--branch',kind],gpus=[gpu],minimum=6500 if isft else 10000,proof=base/'evaluations'/kind/'results.json')
  return {'branch':kind,'state':'evaluated'}
 except Exception as error:
  state['samples'][label].setdefault('branch_failures',{})[kind]=str(error);persist();return {'branch':kind,'state':'failed','reason':str(error)}
async def person(c):
 label=c['sample_id']
 if state['samples'].get(label,{}).get('state')=='terminal_versions_review_pending':return False
 if (R/'samples'/label/'generation/research/generated_validation.json').exists() and state['samples'].get(label,{}).get('state')=='frontend_failed_implementation_or_input_review_required':
  state['samples'][label].update(state='queued',recovery='Validator fixed to compare numeric physical K,c2w,size and coordinate identity; conditioning-only fields are not expected in generated schema; no regeneration.')
 if state['samples'].get(label,{}).get('state','').startswith('frontend_'):
  errors='\n'.join(Path(v['log']).read_text(errors='replace')[-4000:] for k,v in state['jobs'].items() if k.startswith(label+'/') and v.get('state')=='failed' and v.get('log'))
  hard=any(x in errors for x in ['Annotated person clipped by crop','Required crop exceeds source','Exposure span shorter than fixed 25fps window','Illegal exposure-time continuity','Global/incam SMPL geometry is not a consistent rigid pair','Empty foreground: no legal common RGBA','Common frontend foreground gate failed'])
  state['samples'][label].update(state='frontend_hard_failure' if hard else 'frontend_failed_implementation_or_input_review_required',reserve_eligible=hard);persist();return hard
 state['samples'].setdefault(label,{}).update(state='queued',role=c['cohort'],identity_key=c['identity_key']);persist()
 async with case_limit:
  try:
   async with disk_budget(f'{label}/shared_input',.65 if c['output_slots']==61 else 1.1):await frontend(c)
  except Exception as error:
   src=R/'samples'/label/'source';mask=src/'research/batch_gate_mask.json';hard=mask.exists() and json.loads(mask.read_text()).get('passed') is False
   errors='\n'.join(Path(v['log']).read_text(errors='replace')[-4000:] for k,v in state['jobs'].items() if k.startswith(label+'/') and v.get('state')=='failed' and v.get('log'))
   hard=hard or any(x in errors for x in ['Annotated person clipped by crop','Required crop exceeds source','Exposure span shorter than fixed 25fps window','Illegal exposure-time continuity','Global/incam SMPL geometry is not a consistent rigid pair','Empty foreground: no legal common RGBA','Source gate rejected input'])
   state['samples'][label].update(state='frontend_hard_failure' if hard else 'frontend_failed_implementation_or_input_review_required',reason=str(error),reserve_eligible=hard);persist();return hard
  try:
   await stage(label,'ftgs_initialization','ftgs','export_sequence.py',proof=R/'samples'/label/'data/init.npz')
  except Exception as error:
   state['samples'][label].update(state='shared_backend_initialization_failed',reason=str(error),reserve_eligible=False);persist();return False
  try:await stage(label,'4dgs_initialization','ftgs','export_4dgs.py',['--label',label,'--data',R/'samples'/label/'data'],proof=R/'four_dg/data'/label/'fdanyone.json')
  except Exception as error:
   state['samples'][label].update(fourdgs_init_error=str(error));persist()
  state['samples'][label]['state']='training_branches';persist()
  results=await asyncio.gather(*map(lambda k:branch(c,k),['ftgs_base5k','ftgs_opt15k','4dgs_grid25','4dgs_grid61']))
  await stage(label,'comparison','front','comparison.py',['--label',label])
  state['samples'][label].update(state='terminal_versions_review_pending',branches=results);persist();return False
async def heartbeat():
 while True:
  persist();await asyncio.sleep(120)
async def main():
 own=(R/'controller.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);state['state']='active'
 for c in selection['main']+selection['reserve']:
  for b in ['ftgs_base5k','ftgs_opt15k','4dgs_grid25','4dgs_grid61']:
   state['jobs'].setdefault(c['sample_id']+'/'+b,{'state':'preregistered_waiting_dependencies' if c['cohort']=='main' else 'preregistered_inactive_reserve','dependency':'shared24viewRGBA_and_training_only_initialization','cohort':c['cohort']})
 persist();pulse=asyncio.create_task(heartbeat())
 mains=selection['main'];reserves=selection['reserve'];audits=asyncio.gather(*map(audit,mains))
 # Bounded people run independently; the per-case limit avoids multiple large datasets with current disk headroom.
 results=await asyncio.gather(*map(person,mains),return_exceptions=True);await audits
 needed=sum(x is True for x in results)
 if needed:
  pending=list(reserves);activated=[]
  while needed and pending:
   c=pending.pop(0);activated.append(c['sample_id']);result=await person(c)
   if result is not True:needed-=1
  state['activated_reserves']=activated
 state['state']='all_registered_tasks_terminal_review_pending';state['person_exceptions']=[str(x) for x in results if isinstance(x,BaseException)];persist();pulse.cancel()
if __name__=='__main__':asyncio.run(main())
