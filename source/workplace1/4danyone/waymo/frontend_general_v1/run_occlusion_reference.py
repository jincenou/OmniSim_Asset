from pathlib import Path
import asyncio,os,json,time
G=Path(__file__).resolve().parent;O=G/'occlusion_v1';PY='/mnt/ojc/miniconda3/envs/4danyone/bin/python'
async def job(sample,gpu):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',FDANYONE_SAMPLE_ROOT=str(O/'evaluation_reference'/sample),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 with (O/f'reference_{sample}.log').open('x') as log:
  p=await asyncio.create_subprocess_exec(PY,str(G/'extract_occlusion_reference.py'),sample,env=env,stdout=log,stderr=asyncio.subprocess.STDOUT);rc=await p.wait()
 return {'sample':sample,'gpu':gpu,'pid':p.pid,'exit_code':rc,'finished':time.time()}
async def main():
 result=await asyncio.gather(job('P24',0),job('P36',7));(O/'reference_status.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':asyncio.run(main())
