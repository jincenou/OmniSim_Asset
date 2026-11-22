from pathlib import Path
import asyncio,os,json,time,hashlib
G=Path(__file__).resolve().parent;PY='/mnt/ojc/miniconda3/envs/4danyone/bin/python';STATE={'state':'running','jobs':{}}
def update():
 p=G/'dev6_geometry_status.json';t=p.with_suffix('.writing');t.write_text(json.dumps(STATE,indent=2)+'\n');t.replace(p)
async def job(sample,gpu):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 env['FDANYONE_RUNTIME']=str(G.parent/'round2/moving61_runtime' if sample not in ['P04','P30'] else G.parents[1]/'repo')
 try:
  p=await asyncio.create_subprocess_exec('nvidia-smi',f'--id={gpu}','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE);out,_=await p.communicate();assert p.returncode==0
  uuid,free=out.decode().strip().split(',');assert int(free)>4000;env['FD_EXPECTED_GPU_UUID']=uuid.strip()
  with (G/'logs'/f'{sample}_dev6_geometry.log').open('x') as log:
   p=await asyncio.create_subprocess_exec(PY,str(G/'build_dev6_geometry.py'),sample,env=env,stdout=log,stderr=asyncio.subprocess.STDOUT);STATE['jobs'][sample]={'state':'running','pid':p.pid,'gpu':gpu,'started':time.time()};update();rc=await p.wait()
  STATE['jobs'][sample].update(state='complete_needs_review' if rc==0 else 'failed',exit_code=rc,finished=time.time());update()
 except Exception as error:STATE['jobs'][sample]={'state':'failed','gpu':gpu,'error':repr(error)};update()
async def main():
 if (G/'dev6_geometry_status.json').exists():raise FileExistsError('Existing run')
 files=['build_dev6_geometry.py','reliable_reference.py','run_dev6_geometry.py','dev6_geometry_protocol.json'];(G/'dev6_geometry_code_freeze.json').write_text(json.dumps({'before_execution':time.time(),'sha256':{n:hashlib.sha256((G/n).read_bytes()).hexdigest() for n in files}},indent=2)+'\n')
 await asyncio.gather(*(job(sample,gpu) for sample,gpu in zip(['P18','P08','P36','P04','P30','P24'],[1,2,3,4,7,0],strict=True)))
 STATE['state']='finished_with_failures' if any(j['state']=='failed' for j in STATE['jobs'].values()) else 'finished_needs_review';update()
if __name__=='__main__':asyncio.run(main())
