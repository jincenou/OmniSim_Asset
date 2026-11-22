"""One explicitly bounded evaluation retry after a missing decoder import fix."""
import asyncio,json,os,sys,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R.parents[1]/'round2/scripts'));from resource_lease import lease,wait_memory
async def main():
 label,branch,gpu=sys.argv[1:];gpu=int(gpu);assert gpu in [4,5,6,7];c=json.loads((R/'samples'/label/'candidate.json').read_text());env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',FDANYONE_FRAMES=str(c['output_slots']),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');command=['/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python','-u',str(R/'scripts/evaluate_branch.py'),'--label',label,'--branch',branch];path=R/'research'/f'{label}_{branch}_evaluation_repair.json';log=R/'logs'/f'{label}_{branch}_evaluation_attempt2.log'
 async with lease([gpu]) as handles:
  await wait_memory([gpu],6500)
  if shutil.disk_usage(R).free<35.15*1024**3:raise RuntimeError('Disk admission reserve')
  with log.open('xb') as out:
   p=await asyncio.create_subprocess_exec(*command,env=env,cwd=R,stdout=out,stderr=asyncio.subprocess.STDOUT,pass_fds=handles);record={'state':'running','reason':'Evaluation-only decoder import corrected from unavailable av to existing imageio/ffmpeg; model/checkpoint unchanged; failed files preserved','pid':p.pid,'command':command,'physical_gpu':gpu,'log':str(log)};path.write_text(json.dumps(record,indent=2)+'\n');code=await p.wait();record.update(state='complete' if code==0 else 'failed',exit_code=code);path.write_text(json.dumps(record,indent=2)+'\n')
  if code:raise RuntimeError(f'Evaluation repair exited{code}')
if __name__=='__main__':asyncio.run(main())
