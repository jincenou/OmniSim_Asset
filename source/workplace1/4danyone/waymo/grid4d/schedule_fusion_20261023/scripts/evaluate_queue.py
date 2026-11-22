"""Wait for the verified training controller, then run three bounded GPU evaluations."""
import asyncio,json,os,subprocess,time
from pathlib import Path
STUDY=Path(__file__).resolve().parents[1];GRID=STUDY.parent


async def main():
    controller=int((STUDY/'training_controller.pid').read_text())
    terminal=STUDY/'TRAINING_RESULT.json'
    while not terminal.exists():
        try:os.kill(controller,0)
        except ProcessLookupError:
            await asyncio.sleep(2)
            if not terminal.exists():raise RuntimeError('Training controller exited without terminal result')
        await asyncio.sleep(10)
    result=json.loads(terminal.read_text());assert not result['errors'],result
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    uuids={int(s.split(',')[0]):s.split(',')[1].strip() for s in uuids.splitlines()}
    async def one(label,gpu):
        env=os.environ.copy();toolkit='/mnt/ojc/miniconda3/envs/4dgaussians'
        env.update(CUDA_VISIBLE_DEVICES=uuids[gpu],PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',CUDA_HOME=toolkit,
            CC='/usr/bin/gcc-9',CXX='/usr/bin/g++-9',TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='4',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
            PATH='/mnt/ojc/miniconda3/envs/grid4d/bin:'+toolkit+'/bin:'+env['PATH'],GRID4D_HASH_BUILD=str(GRID/'cache/extensions/hashencoder'),
            TORCH_EXTENSIONS_DIR=str(GRID/'cache/extensions'),TORCH_HOME=str(GRID/'cache/torch'))
        with (STUDY/'logs'/f'{label}_evaluation.log').open('xb') as stream:
            p=await asyncio.create_subprocess_exec('/mnt/ojc/miniconda3/envs/grid4d/bin/python','-B','-u',str(STUDY/'scripts/evaluate.py'),
                '--label',label,env=env,cwd=STUDY,stdout=stream,stderr=asyncio.subprocess.STDOUT)
            code=await p.wait()
        if code:raise RuntimeError(label+' evaluation failed')
        return {'label':label,'physical_gpu':gpu,'state':'passed'}
    results=await asyncio.gather(*(one(label,gpu) for label,gpu in zip(['N04','N06','N11'],[7,0,1])),return_exceptions=True)
    errors=[str(x) for x in results if isinstance(x,BaseException)]
    (STUDY/'EVALUATION_RESULT.json').write_text(json.dumps({'state':'failed' if errors else 'complete_visual_review_pending','results':[x for x in results if not isinstance(x,BaseException)],'errors':errors,'at':time.time()},indent=2)+'\n')
    if errors:raise RuntimeError(errors)


asyncio.run(main())
