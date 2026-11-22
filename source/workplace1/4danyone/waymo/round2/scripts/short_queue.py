import asyncio,fcntl,os,shutil
from after_batch import ROUND,W,read,write,now
from resource_lease import lease,wait_memory
PY='/mnt/ojc/miniconda3/envs/4danyone/bin/python'
async def main():
 with (ROUND/'short_queue.lock').open('a') as own:
  fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
  path=ROUND/'short_queue.json'
  if path.exists():raise FileExistsError(path)
  state={'state':'waiting_for_mobility_evaluation','pid':os.getpid(),'started_at':now(),'stages':[],'gpu':4}
  def save(**fields):state.update(fields,updated_at=now());write(path,state)
  save()
  while True:
   upstream=read(ROUND/'mobility_after.json')
   if upstream['state']=='mobility_metrics_and_videos_complete_review_pending':break
   if upstream['state'] in ('completed_with_failures','evaluation_failed'):raise RuntimeError('Upstream evaluation failed')
   os.kill(upstream['pid'],0);await asyncio.sleep(30)
  if read(ROUND/'short_temporal_preflight.json')['state']!='CPU_contract_checks_passed_GPU_pending':raise RuntimeError('Short contract checks incomplete')
  async with lease([4]) as handles:
   await wait_memory([4],30000)
   if shutil.disk_usage(ROUND).free<20*1024**3:raise RuntimeError('Disk reserve below20GiB')
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',FDANYONE_WAYMO_ROOT=str(ROUND/'experiments/shortS02_v2'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
   env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
   async def stage(name,script,*args):
    cmd=[PY,str(ROUND/'scripts'/script),*args];log=ROUND/'logs'/f'short_v2_{name}.log'
    with log.open('xb') as stream:
     child=await asyncio.create_subprocess_exec(*cmd,cwd=ROUND/'short_runtime',env=env,stdout=stream,stderr=asyncio.subprocess.STDOUT,pass_fds=(*handles,own.fileno()))
     row={'stage':name,'pid':child.pid,'command':cmd,'log':str(log),'state':'running'};state['stages'].append(row);save(state=name);code=await child.wait()
    log.with_suffix('.exit').write_text(str(code)+'\n');row.update(exit_code=code,state='complete' if code==0 else 'failed');save()
    if code:save(state='failed');raise RuntimeError(f'{name} failed, output preserved')
   await stage('generation','short_generate.py','smoke')
   await stage('validation','short_validate.py')
   save(state='short_video_complete_visual_review_pending')
if __name__=='__main__':asyncio.run(main())
