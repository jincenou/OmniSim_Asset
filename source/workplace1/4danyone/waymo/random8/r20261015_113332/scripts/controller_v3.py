"""Persistent, bounded asynchronous queue for the frozen random8 protocol."""
import asyncio,fcntl,hashlib,json,os,shutil,signal,sys,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from contextlib import asynccontextmanager
R=Path(__file__).resolve().parents[1];W=R.parents[1];P=W.parent;S=R/'scripts';OLD=W/'round2';ENV=Path('/mnt/ojc/miniconda3/envs')
sys.path.insert(0,str(OLD/'scripts'));from resource_lease import lease
PY={k:str(ENV/name/'bin/python') for k,name in [('front','4danyone'),('sam','4danyone-sam2'),('ftgs','4danyone-ftgs'),('hust','4dgaussians')]};RESERVE=int(35.75*1024**3)
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
 if proof is not None and proof.exists():
  # Atomic terminal proofs or validated same-input prefixes; retain original exitcodes.
  if name.endswith('_evaluation') or name in ['ftgs_base5k','ftgs_opt15k','4dgs_grid25','4dgs_grid61']:
   prior=json.loads(receipt.read_text()) if receipt.exists() else {}
   repaired=R/'research'/f'{label}_{name}_repair.json'
   if repaired.exists() and json.loads(repaired.read_text()).get('exit_code')==0:prior=json.loads(repaired.read_text())
   state['jobs'][key]={**prior,'state':'complete_existing_terminal_proof_verified','proof':str(proof),'exit_code':prior.get('exit_code')};persist();return
  if name in ['camera_audit','native','interpolate','source_prepare','source_segment','source_composite','motion','motion_gate','conditioning','generate24','rgba','ftgs_initialization','4dgs_initialization','source_media']:
   state['jobs'][key]={'state':'existing_prefix_verified','proof':str(proof),'exit_code':None};persist();return
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

GEN_SEM=asyncio.Semaphore(3);PREP_SEM=asyncio.Semaphore(2);BACK_SEM=asyncio.Semaphore(4);MATERIALIZE=asyncio.Semaphore(1)
terminal_tasks=[]
def alive(pid):
 try:return str(R).encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
 except (FileNotFoundError,PermissionError):return False
async def choose_gpu(minimum):
 while True:
  p=await asyncio.create_subprocess_exec('nvidia-smi','--id=4,5,6,7','--query-gpu=index,memory.free','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE);raw,_=await p.communicate();assert p.returncode==0;devices=[tuple(map(int,row.split(b','))) for row in raw.splitlines()]
  for gpu,free in sorted(devices,key=lambda x:(-x[1],x[0])):
   if free<minimum:continue
   with (OLD/'gpu_locks'/f'gpu{gpu}.lock').open('a') as handle:
    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:continue
   return gpu
  await asyncio.sleep(30)
async def wait_single_profile():
 missing=0
 while True:
  path=R/'SINGLE_GPU_PROFILE_STATUS.json'
  if path.exists():
   j=json.loads(path.read_text())
   if j['state']=='official_single_gpu24_locally_validated_all24_decoded':return
   if j['state']=='failed':raise StageError('SingleGPU24validation failed;no silent execution-profile switch')
   if j.get('pid') and not alive(j['pid']) and j['state'].startswith('running'):
    missing+=1
    if missing>=3:raise StageError('SingleGPU24processgone across3polls; inspect validation rather than restart')
   else:missing=0
  await asyncio.sleep(30)
async def prepare(c):
 label=c['sample_id'];base=R/'samples'/label;src=base/'source';gen=base/'generation'
 if (gen/'preflight/review.json').exists():
  assert json.loads((gen/'preflight/review.json').read_text())['allow_generation'];return
 repairs=R/'FRONTEND_REPAIR_STATUS.json'
 if repairs.exists() and label in json.loads(repairs.read_text())['cases']:
  while True:
   j=json.loads(repairs.read_text())['cases'][label]
   if j['state']=='same_input_repaired_ready_for_first24view_generation':return
   if j['state']=='failed_after_bounded_repair':raise StageError('Unchanged frontendgate failed after bounded engineeringrepair:'+j['error'])
   await asyncio.sleep(30)
 gate=src/'research/batch_gate_mask.json'
 if gate.exists() and not json.loads(gate.read_text())['passed']:raise StageError('Unchanged frontend foreground hard gate;no output-based threshold relaxation')
 async with PREP_SEM:
  async with disk_budget(label+'/prepare_source',.12):
   await stage(label,'camera_audit','hust','raw_camera_audit.py',['--review-id',label,'--manifest',R/'selection_manifest.json'],proof=R/'camera_audit'/label/'audit.json')
   await stage(label,'native','front','m02_prepare.py',proof=src/'native.json')
   gpu=await choose_gpu(6500);await stage(label,'interpolate','front','interpolate.py',gpus=[gpu],minimum=6500,proof=src/'inputs_rife/frame_mapping.json' if (src/'inputs_rife').exists() else src/'inputs/frame_mapping.json')
   await stage(label,'source_prepare','front','source_foreground.py',['prepare'],proof=src/'segmentation/frames/00000.jpg')
   gpu=await choose_gpu(7000);await stage(label,'source_segment','sam','source_foreground.py',['segment'],gpus=[gpu],minimum=7000,proof=src/'segmentation/target_masks.npz')
   await stage(label,'source_composite','front','source_foreground.py',['composite'],proof=src/'research/batch_gate_mask.json');assert json.loads((src/'research/batch_gate_mask.json').read_text())['passed']
   gpu=await choose_gpu(15000);await stage(label,'motion','front','motion.py',gpus=[gpu],minimum=15000,proof=src/'motion_pair/calibrated_moving/motion.safetensors')
   await stage(label,'motion_gate','front','source_foreground.py',['motion_gate'],proof=src/'research/motion_gate.json')
   gpu=await choose_gpu(9000);await stage(label,'conditioning','front','conditioning.py',gpus=[gpu],minimum=9000,proof=gen/'preflight/review.json')
async def generated(c):
 label=c['sample_id'];base=R/'samples'/label;gen=base/'generation';proof=gen/'research/generated_validation.json';state['samples'].setdefault(label,{}).update(role=c['cohort'],identity_key=c['identity_key'],state='preparing_or_recovering_same_frozen_input');persist()
 try:
  await prepare(c)
  if not proof.exists():
   await wait_single_profile()
   async with GEN_SEM:
    if label=='N01':assert proof.exists(),'N01validation must finish without regenerating it'
    else:
     async with disk_budget(label+'/generate24',.15):
      gpu=await choose_gpu(32000);await stage(label,'generate24','front','generate_single.py',['turbo'],gpus=[gpu],minimum=32000,proof=gen/'outputs/turbo/metadata.json')
      from finish_generation import finish
      await asyncio.to_thread(finish,base,c['output_slots'])
  assert json.loads(proof.read_text())['state']=='all24_videos_decoded_same_retained_cameras'
  await stage(label,'source_media','front','source_media.py',['--label',label],proof=base/'source/visualization/manifest.json')
  state['samples'][label].update(state='legal_shared24view_generated_backend_pending',frontend_protocol='v1.1_frozen' if label=='N04' else 'random8-frontend-v1.2',execution_profile='original4GPU' if label=='N04' else 'official_single_gpu24');persist();terminal_tasks.append(asyncio.create_task(reconstruct(c)));return True
 except Exception as e:
  errors=str(e)+'\n'+'\n'.join(Path(v['log']).read_text(errors='replace')[-5000:] for k,v in state['jobs'].items() if k.startswith(label+'/') and v.get('state')=='failed' and v.get('log'))
  hard=any(x in errors for x in ['Unchanged frontendgate failed','Unchanged frontend foreground hard gate','Common frontend foreground gate failed','Annotated person clipped by crop','Exposure interval lacks coverage','Illegal exposure-time continuity','Global/incam SMPL geometry is not a consistent rigid pair','Empty foreground: no legal common RGBA'])
  state['samples'][label].update(state='frontend_hard_failure_kept_in_denominator' if hard else 'blocked_frontend_infrastructure_review_required',reason=f'{type(e).__name__}:{e}',reserve_eligible=hard);persist();return False if hard else None
async def train_branch(c,kind):
 label=c['sample_id'];base=R/'samples'/label;isft=kind.startswith('ftgs');modelpath=base/kind if isft else R/'four_dg/runs'/f'{label}_grid{kind[-2:]}';trained=modelpath/('training_complete.json' if isft else 'complete.json');evaluation=base/'evaluations'/kind/'results.json'
 try:
  if not trained.exists():
   async with disk_budget(label+'/'+kind,.35 if kind=='ftgs_base5k' else .6 if isft else 1.65):
    async with BACK_SEM:
     gpu=await choose_gpu(6500 if isft else 12000);await stage(label,kind,'ftgs' if isft else 'hust','train_ftgs.py' if isft else 'train_4dgs.py',['--label',label,'--branch',kind] if isft else ['--label',label,'--grid',kind[-2:]],gpus=[gpu],minimum=6500 if isft else 12000,proof=trained)
  else:await stage(label,kind,'ftgs' if isft else 'hust','train_ftgs.py' if isft else 'train_4dgs.py',proof=trained)
  if not evaluation.exists():
   # A failed earlier evaluation requires an explicit repair,never an unbounded automatic retry.
   prior=R/'logs'/f'{label}_{kind}_evaluation.json'
   if prior.exists() and json.loads(prior.read_text()).get('state')=='failed':raise StageError('Existing evaluation failed;explicit review/repair required')
   async with disk_budget(label+'/'+kind+'_evaluation',.08):
    gpu=await choose_gpu(6500 if isft else 10000);await stage(label,kind+'_evaluation','ftgs' if isft else 'hust','evaluate_branch.py',['--label',label,'--branch',kind],gpus=[gpu],minimum=6500 if isft else 10000,proof=evaluation)
  else:await stage(label,kind+'_evaluation','ftgs' if isft else 'hust','evaluate_branch.py',proof=evaluation)
  return {'branch':kind,'state':'evaluated'}
 except Exception as e:
  state['jobs'][label+'/'+kind].update(state='failed_backend_no_person_replacement',reason=f'{type(e).__name__}:{e}');persist();return {'branch':kind,'state':'failed','reason':str(e)}
async def reconstruct(c):
 label=c['sample_id'];base=R/'samples'/label
 async with MATERIALIZE:
  try:
   maskproof=R/'research'/f'{label}-mask-complete.json'
   if not maskproof.exists():
    async with disk_budget(label+'/shared_RGBA',.4 if c['output_slots']==61 else .8):
     gpu=await choose_gpu(10000);await stage(label,'rgba','sam','full_masks.py',gpus=[gpu],minimum=10000,proof=maskproof)
   await stage(label,'ftgs_initialization','ftgs','export_sequence.py',proof=base/'data/init.npz')
   await stage(label,'4dgs_initialization','ftgs','export_4dgs.py',['--label',label,'--data',base/'data'],proof=R/'four_dg/data'/label/'fdanyone.json')
   state['samples'][label]['state']='training_branches';persist();branches=await asyncio.gather(*map(lambda b:train_branch(c,b),['ftgs_base5k','ftgs_opt15k','4dgs_grid25','4dgs_grid61']))
   await stage(label,'comparison','front','comparison.py',['--label',label])
   state['samples'][label].update(state='terminal_versions_review_pending',branches=branches);persist()
  except Exception as e:state['samples'][label].update(state='backend_preparation_failed_no_person_replacement',reason=f'{type(e).__name__}:{e}');persist()
async def reserves(index,needed):
 if needed<=0 or index>=len(selection['reserve']):return
 c=selection['reserve'][index];ok=await generated(c);await reserves(index+1,needed-(1 if ok is True else 0))
async def heartbeat():
 while True:
  persist();await asyncio.sleep(120)

async def main():
 own=(R/'controller.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);state.update(state='active_controller_v2',controller_pid=os.getpid(),frontend_repair_protocol='v1.2',single_gpu_generation_profile='official_single_gpu24',no_resume_or_restart_failed_training=True);persist();pulse=asyncio.create_task(heartbeat())
 result=await asyncio.gather(*map(generated,selection['main']));await reserves(0,sum(x is False for x in result));await asyncio.gather(*terminal_tasks);state.update(state='all_registered_tasks_terminal_v2_review_pending',infrastructure_blocked_tracks=sum(x is None for x in result),legal_generated_tracks=sum(v.get('state') in ['terminal_versions_review_pending','backend_preparation_failed_no_person_replacement'] for v in state['samples'].values()));persist();pulse.cancel()
 from report import main as report
 report()
if __name__=='__main__':asyncio.run(main())
