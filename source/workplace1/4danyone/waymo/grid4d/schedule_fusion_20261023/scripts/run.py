"""Bounded async training DAG; one added job per authorized physical GPU."""
import asyncio,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
STUDY=Path(__file__).resolve().parents[1]
GRID=STUDY.parent
STATES={}


def record():
    tmp=STUDY/'STATUS.writing';tmp.write_text(json.dumps({'updated_at':time.time(),'jobs':STATES},indent=2)+'\n');tmp.replace(STUDY/'STATUS.json')


async def main():
    plan=json.loads((STUDY/'PLAN.json').read_text())
    frozen=json.loads((STUDY/'CODE_FREEZE.json').read_text())
    for name,wanted in frozen['files'].items():assert hashlib.sha256((STUDY/name).read_bytes()).hexdigest()==wanted,name
    identities=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    identities={int(line.split(',')[0]):line.split(',')[1].strip() for line in identities.splitlines()}
    slots=asyncio.Queue()
    for gpu in [7,0,1,2,3,4,5,6]:slots.put_nowait(gpu)

    async def job(label,arm,source=None):
        gpu=await slots.get();key=label+'_'+arm
        try:
            assert shutil.disk_usage(STUDY).free>3*2**30,'Insufficient space for next checkpoint'
            toolkit='/mnt/ojc/miniconda3/envs/4dgaussians';env=os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=identities[gpu],PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
                CUDA_HOME=toolkit,CC='/usr/bin/gcc-9',CXX='/usr/bin/g++-9',TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='4',
                OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                PATH='/mnt/ojc/miniconda3/envs/grid4d/bin:'+toolkit+'/bin:'+env['PATH'],
                GRID4D_HASH_BUILD=str(GRID/'cache/extensions/hashencoder'),TORCH_EXTENSIONS_DIR=str(GRID/'cache/extensions'),
                TORCH_HOME=str(GRID/'cache/torch'),MPLCONFIGDIR=str(GRID/'cache/matplotlib'))
            command=['/mnt/ojc/miniconda3/envs/grid4d/bin/python','-B','-u',str(STUDY/'scripts/train.py'),'--label',label,'--arm',arm]
            if source:command+=['--source',str(source)]
            STATES[key]={'state':'running','physical_gpu':gpu,'uuid':identities[gpu],'started_at':time.time()};record()
            with (STUDY/'logs'/f'{key}.log').open('xb') as stream:
                p=await asyncio.create_subprocess_exec(*command,env=env,cwd=STUDY,stdout=stream,stderr=asyncio.subprocess.STDOUT)
                STATES[key]['pid']=p.pid;record();code=await p.wait()
            STATES[key].update(state='completed' if code==0 else 'failed',exit_code=code,finished_at=time.time());record()
            if code:raise RuntimeError(key+' failed; see its log')
            return STUDY/'runs'/key
        finally:slots.put_nowait(gpu)

    async def subject(label):
        prefix=await job(label,'prefix')
        async def late():
            middle=await job(label,'middle',prefix)
            return await asyncio.gather(job(label,'F0',middle),job(label,'F1',middle))
        return await asyncio.gather(late(),job(label,'F2',prefix))
    results=await asyncio.gather(*(subject(label) for label in plan['subjects']),return_exceptions=True)
    errors=[str(r) for r in results if isinstance(r,BaseException)]
    (STUDY/'TRAINING_RESULT.json').write_text(json.dumps({'state':'failed' if errors else 'training_complete_evaluation_pending','errors':errors,'at':time.time()},indent=2)+'\n')
    if errors:raise RuntimeError(errors)


if __name__=='__main__':asyncio.run(main())
