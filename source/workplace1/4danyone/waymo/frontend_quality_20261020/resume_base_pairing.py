"""Resume only proven pairing-schema failures; do not restart live jobs."""
import asyncio,fcntl,json,os,time
from pathlib import Path
import run_base_ablation as base
ROOT=base.ROOT;STATE={'state':'running','jobs':{}}
def save():base.dump(ROOT/'base_resume_status.json',STATE)
async def resume(case,gpu):
 old=base.RUN/'scenes'/case;name=old.name;work=ROOT/'base_ablation'/name;gen=work/'generation';source=work/'source'
 status=json.loads((ROOT/'base_status.json').read_text())['jobs'][name]
 if status.get('state')!='failed' or status.get('error')!="RuntimeError('pairing exit 1')":return
 log=(ROOT/'logs'/f'{name}_base_pairing.log').read_text()
 assert "KeyError: 'world_to_camera'" in log
 assert not (gen/'outputs/base').exists()
 lease=(base.W/'gpu_leases_20261019'/f'gpu{gpu}.lock').open('a')
 try:
  fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
  count=json.loads((old/'generation.json').read_text())['output']['frames_per_video']
  env=os.environ.copy();env.pop('PYTHONPATH',None)
  env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
   FDANYONE_BASELINE=str(old),FDANYONE_WAYMO_ROOT=str(gen),FDANYONE_SOURCE_EXPERIMENT=str(source),FDANYONE_SAMPLE_ROOT=str(source),FDANYONE_FRAMES=str(count),
   FDANYONE_RUNTIME=str(base.W/'round2/moving61_runtime' if count==61 else base.W.parent/'repo'),FDANYONE_RETAINED_VIEWS='24',
   TORCH_EXTENSIONS_DIR=str(base.RUN/'cache/torch_extensions'),TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='4',TMPDIR=str(ROOT/'tmp'))
  for stage,script,args in [('pairing',ROOT/'check_conditioning.py',[]),('generate',ROOT/'generate_base.py',['base'])]:
   with (ROOT/'logs'/f'{name}_base_{stage}_resume.log').open('x') as output:
    begin=time.time();p=await asyncio.create_subprocess_exec(base.PY,str(script),*args,env=env,stdout=output,stderr=output)
    STATE['jobs'][name]={'state':'running','stage':stage,'gpu':gpu,'pid':p.pid,'started':begin};save();rc=await p.wait()
    base.dump(ROOT/'receipts'/f'{name}_base_{stage}_resume.json',dict(exit_code=rc,pid=p.pid,gpu=gpu,seconds=time.time()-begin,script_sha256=base.sha(script)))
    if rc:raise RuntimeError(f'{stage} exit {rc}')
  STATE['jobs'][name].update(state='generated_needs_validation',finished=time.time());save()
 except Exception as error:STATE['jobs'][name]={'state':'failed','error':repr(error),'gpu':gpu};save()
 finally:fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
async def main():
 if (ROOT/'base_resume_status.json').exists():raise FileExistsError('Inspect existing recovery before another run')
 save();await asyncio.gather(*(resume(case,gpu) for gpu,case in enumerate(base.CASES)));STATE['state']='finished';save()
if __name__=='__main__':asyncio.run(main())
