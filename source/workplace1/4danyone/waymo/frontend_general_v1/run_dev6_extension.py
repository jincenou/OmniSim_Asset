from pathlib import Path
import asyncio,os,json,time
G=Path(__file__).resolve().parent;PY='/mnt/ojc/miniconda3/envs/4danyone/bin/python'
async def job(sample):
 env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
 with (G/'logs'/f'{sample}_dev6_extension.log').open('x') as log:
  p=await asyncio.create_subprocess_exec(PY,str(G/'optimize_motion.py'),sample,'--variant','local_temporal','--extend-dev6',env=env,stdout=log,stderr=asyncio.subprocess.STDOUT);rc=await p.wait()
 return {'sample':sample,'pid':p.pid,'exit_code':rc,'finished':time.time()}
async def main():
 r=await asyncio.gather(job('P04'),job('P24'));(G/'dev6_extension_status.json').write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':asyncio.run(main())
