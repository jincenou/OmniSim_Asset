"""Registered33k decision -> at most one shared5k extension -> final analysis."""
import asyncio,json,os,subprocess,time
from pathlib import Path
STUDY=Path(__file__).resolve().parents[1];GRID=STUDY.parent


async def process(command,log,env=None):
    with log.open('xb') as f:
        p=await asyncio.create_subprocess_exec(*command,env=env,cwd=STUDY,stdout=f,stderr=asyncio.subprocess.STDOUT)
        code=await p.wait()
    if code:raise RuntimeError(f'{log.name}: exit{code}')


async def main():
    terminal=STUDY/'EVALUATION_RESULT.json';pid=int((STUDY/'evaluation_controller.pid').read_text())
    while not terminal.exists():
        try:os.kill(pid,0)
        except ProcessLookupError:
            await asyncio.sleep(2)
            if not terminal.exists():raise RuntimeError('Evaluation controller missing without terminal result')
        await asyncio.sleep(10)
    data=json.loads(terminal.read_text());assert not data['errors'],data
    py='/mnt/ojc/miniconda3/envs/4danyone/bin/python'
    await process([py,'-B',str(STUDY/'scripts/analyze.py')],STUDY/'logs/analyze33.log')
    decision=json.loads((STUDY/'DECISION_33K.json').read_text());arm=decision['selected_for_extension']
    if arm:
        lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()
        uuids={int(s.split(',')[0]):s.split(',')[1].strip() for s in lines}
        async def subject(label,gpu):
            toolkit='/mnt/ojc/miniconda3/envs/4dgaussians';env=os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=uuids[gpu],PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',CUDA_HOME=toolkit,
                CC='/usr/bin/gcc-9',CXX='/usr/bin/g++-9',TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='4',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                PATH='/mnt/ojc/miniconda3/envs/grid4d/bin:'+toolkit+'/bin:'+env['PATH'],GRID4D_HASH_BUILD=str(GRID/'cache/extensions/hashencoder'),
                TORCH_EXTENSIONS_DIR=str(GRID/'cache/extensions'),TORCH_HOME=str(GRID/'cache/torch'))
            backend='/mnt/ojc/miniconda3/envs/grid4d/bin/python'
            begin=time.monotonic()
            await process([backend,'-B','-u',str(STUDY/'scripts/extend.py'),'--label',label,'--arm',arm],STUDY/'logs'/f'{label}_{arm}38_train.log',env)
            await process([backend,'-B','-u',str(STUDY/'scripts/evaluate.py'),'--label',label,'--methods','F0','F1','F2',arm+'38'],STUDY/'logs'/f'{label}_extension_evaluation.log',env)
            return {'label':label,'method':arm+'38','gpu':gpu,'seconds_including_evaluation':time.monotonic()-begin}
        results=await asyncio.gather(*(subject(label,gpu) for label,gpu in zip(['N04','N06','N11'],[7,0,1])),return_exceptions=True)
        errors=[str(x) for x in results if isinstance(x,BaseException)]
        (STUDY/'EXTENSION_RESULT.json').write_text(json.dumps({'state':'failed' if errors else 'complete','errors':errors,'results':[x for x in results if not isinstance(x,BaseException)]},indent=2)+'\n')
        if errors:raise RuntimeError(errors)
    await process([py,'-B',str(STUDY/'scripts/analyze.py'),'--final'],STUDY/'logs/analyze_final.log')
    (STUDY/'NUMERICAL_COMPLETE.json').write_text(json.dumps({'state':'fixed_numeric_complete_visual_review_pending','extension':arm,'at':time.time()},indent=2)+'\n')


asyncio.run(main())
