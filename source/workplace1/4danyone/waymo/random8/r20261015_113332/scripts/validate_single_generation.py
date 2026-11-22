"""First,explicitly documented validation of the official single-card24view path."""
import asyncio,fcntl,json,os,sys,shutil,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R.parents[1]/'round2/scripts'));from resource_lease import lease,wait_memory
from frontend_repairs import env
from finish_generation import finish
PATH=R/'SINGLE_GPU_PROFILE_STATUS.json'
def save(v):
 t=PATH.with_suffix('.writing');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(PATH)
async def main():
 lock=(R/'single_gpu_profile.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);label='N01';assert not (R/'samples'/label/'generation/outputs/turbo').exists();ready=json.loads((R/'samples'/label/'generation/preflight/review.json').read_text());assert ready['allow_generation'];assert shutil.disk_usage(R).free>35.95*1024**3
 async with lease([6]) as handles:
  await wait_memory([6],32000);e=env(label);e['CUDA_VISIBLE_DEVICES']='6';e['FDANYONE_RETAINED_VIEWS']='24';command=['/mnt/ojc/miniconda3/envs/4danyone/bin/python','-u',str(R/'scripts/generate_single.py'),'turbo'];log=R/'logs/N01_generate24_single_attempt1.log'
  with log.open('xb') as stream:
   child=await asyncio.create_subprocess_exec(*command,env=e,cwd=R,stdout=stream,stderr=asyncio.subprocess.STDOUT,pass_fds=handles,start_new_session=True);record={'state':'running_single_gpu24_validation','pid':child.pid,'process_group':child.pid,'label':label,'physical_gpus':[6],'command':command,'environment':{k:v for k,v in e.items() if k.startswith(('FDANYONE_','CUDA_'))},'log':str(log),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()};save(record);code=await child.wait()
  record.update(exit_code=code,ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),state='failed' if code else 'generated_validation_pending');save(record)
  if code:raise RuntimeError(f'Official single24 validation failed{code};do not change inputs or retryunbounded')
 finish(R/'samples'/label,61);record.update(state='official_single_gpu24_locally_validated_all24_decoded',validation=str(R/'samples'/label/'generation/research/generated_validation.json'));save(record);profile=R/'research/single_gpu24_profile.json';p=json.loads(profile.read_text());p['local_validation']='passed N01 full24view F61 inference,matching retained physical cameras,allvideosdecoded';profile.write_text(json.dumps(p,indent=2)+'\n')
if __name__=='__main__':asyncio.run(main())
