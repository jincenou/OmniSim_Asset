"""Six predeclared real-RGB counterfactual PoseNet proposals and equal-budget alignment."""
from pathlib import Path
import asyncio,os,json,time,hashlib,shutil
G=Path(__file__).resolve().parent;O=G/'occlusion_v1';PY='/mnt/ojc/miniconda3/envs/4danyone/bin/python'
STATE={'state':'running','jobs':{}};CPU=asyncio.Semaphore(2)
def save(p,v):
 t=p.with_suffix('.writing');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def update():STATE['updated_at']=time.time();save(O/'rohm_status.json',STATE)
async def job(item,gpu):
 root=Path(item['root']);name=root.parent.name;sample=item['sample'];cfg=root/'fit_config.json'
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 try:
  probe=await asyncio.create_subprocess_exec('nvidia-smi',f'--id={gpu}','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE)
  data,_=await probe.communicate()
  if probe.returncode:raise RuntimeError('Device query failed')
  uuid,free=data.decode().strip().split(',');env['FD_EXPECTED_GPU_UUID']=uuid.strip()
  if int(free)<4000:raise RuntimeError('Insufficient4GiB shared headroom')
  async def execute(stage,script,*extra):
   if shutil.disk_usage(G).free<5*2**30:raise RuntimeError('Insufficient safe write space')
   with (root/f'{stage}.log').open('x') as log:
    p=await asyncio.create_subprocess_exec(PY,str(G/script),sample,'--case-config',str(cfg),*extra,env=env,stdout=log,stderr=asyncio.subprocess.STDOUT)
    STATE['jobs'][name]={'state':'running','stage':stage,'pid':p.pid,'gpu':gpu,'started':time.time()};update();rc=await p.wait()
    save(root/f'{stage}_receipt.json',{'pid':p.pid,'exit_code':rc,'finished':time.time()})
    if rc:raise RuntimeError(f'{stage} failed{rc}')
  async with CPU:await execute('rohm_prepare','prepare_rohm.py')
  await execute('rohm_sample','sample_rohm.py')
  async with CPU:await execute('rohm_align','optimize_motion.py','--variant','rohm')
  STATE['jobs'][name].update(state='complete_needs_hidden_evaluation',finished=time.time());update()
 except Exception as e:STATE['jobs'][name]={'state':'failed','error':repr(e),'gpu':gpu};update()
async def main():
 if (O/'rohm_status.json').exists():raise FileExistsError('Inspect existing queue before rerun')
 files=['prepare_rohm.py','sample_rohm.py','optimize_motion.py','run_occlusion_rohm.py','gap_proposal.py','camera_geometry.py']
 save(O/'rohm_implementation_freeze.json',{'before_execution_utc_seconds':time.time(),'sha256':{n:hashlib.sha256((G/n).read_bytes()).hexdigest() for n in files}})
 jobs=[x for x in json.loads((O/'jobs.json').read_text()) if x['branch']=='rgb']
 await asyncio.gather(*(job(item,gpu) for item,gpu in zip(jobs,[1,2,3,4,7,0],strict=True)))
 STATE['state']='finished_with_failures' if any(x['state']=='failed' for x in STATE['jobs'].values()) else 'finished_needs_evaluation';update()
if __name__=='__main__':asyncio.run(main())
