"""Bounded async orchestration; each independent experiment has its own GPU."""
import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

BASE=Path(__file__).resolve().parents[1]
PY='/mnt/ojc/miniconda3/envs/4danyone/bin/python'
SAM='/mnt/ojc/miniconda3/envs/4danyone-sam2/bin/python'


async def case(name,stage):
    root=BASE/'experiments'/name;config=json.loads((root/'config.json').read_text())
    env=os.environ.copy();env.update(FDANYONE_WAYMO_ROOT=str(root),FDANYONE_RIFE_SOURCE=str(root/'inputs_native/frame_mapping.json'),CUDA_VISIBLE_DEVICES=str(config['gpu']))
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TORCH_CUDNN_V8_API_DISABLED='1',CUDNN_FRONTEND_DISABLE='1',NVIDIA_TF32_OVERRIDE='0')
    env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    if stage=='turbo':env['CUDA_VISIBLE_DEVICES']='4,5,6,7'
    async def run(script,*args,python=PY):
        log=root/'logs'/f"{stage}-{Path(script).stem}{'-'+Path(args[0]).name if args else ''}.log"
        with log.open('w') as out:
            p=await asyncio.create_subprocess_exec(python,str(BASE/'scripts'/script),*args,env=env,cwd=BASE.parent/'repo',stdout=out,stderr=asyncio.subprocess.STDOUT)
            code=await p.wait()
        log.with_suffix('.exit').write_text(str(code)+'\n')
        if code:raise RuntimeError(f'{name} {script} {args}: exit {code}; see {log}')
    state={'experiment':name,'stage':stage,'state':'running','physical_gpu_ids':[int(x) for x in env['CUDA_VISIBLE_DEVICES'].split(',')],'started_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}
    path=root/'experiment_status.json';path.write_text(json.dumps(state,indent=2)+'\n')
    try:
        if stage=='prepare':
            await run('experiment_prepare.py','native')
            await run('interpolate.py')
            await run('experiment_prepare.py','frames')
            await run('segment_target.py',python=SAM)
            await run('prepare_foreground.py')
            await run('validate_input.py')
            await run('experiment_review.py')
            await run('preflight.py')
            state['state']='preflight_visual_review_pending'
        elif stage=='preflight':
            await run('preflight.py')
            state['state']='preflight_visual_review_pending'
        elif stage=='refine':
            await run('segment_target.py',python=SAM)
            await run('prepare_foreground.py')
            await run('validate_input.py')
            await run('experiment_review.py')
            await run('preflight.py')
            state['state']='preflight_visual_review_pending'
        elif stage in ('smoke','turbo'):
            await run('generate.py',stage)
            await run('../../scripts/visualize.py',str(root/'outputs'/stage))
            await run('validate_output.py',stage)
            state['state']='generated_visual_review_pending'
        else:raise ValueError(stage)
    except Exception as error:
        state.update(state='failed',error=str(error));print(str(error),flush=True)
    finally:
        state['updated_at']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat();path.write_text(json.dumps(state,indent=2)+'\n')
    return state


async def main():
    stage=sys.argv[1];names=sys.argv[2:]
    assert 1<=len(names)<=4
    results=await asyncio.gather(*map(lambda name:case(name,stage),names))
    (BASE/'experiments'/f'{stage}-results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results),flush=True)
    if any(x['state']=='failed' for x in results):sys.exit(1)


if __name__=='__main__':asyncio.run(main())
