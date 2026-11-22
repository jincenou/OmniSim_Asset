"""One explicit 21-frame experiment; no change to the long-sequence batch."""
import asyncio
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

S=Path(__file__).resolve().parents[1]
ROOT=S/'S02_21frames'
PY='/mnt/ojc/miniconda3/envs/4danyone/bin/python'
SAM='/mnt/ojc/miniconda3/envs/4danyone-sam2/bin/python'


def save(value):
    value['updated_at']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    path=ROOT/'run.json';temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


async def main(fd):
    if (ROOT/'run.json').exists():raise FileExistsError('Inspect prior stage records before explicitly resuming this experiment')
    state={'state':'waiting_for_gpu','pid':os.getpid(),'frames':21,'observed_frames':9,'span_seconds':.8,'done':['native']}
    save(state)
    while True:
        p=await asyncio.create_subprocess_exec('nvidia-smi','--id=4,5,6,7','--query-gpu=index,memory.free',
            '--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE)
        data,_=await p.communicate()
        if p.returncode:raise RuntimeError('GPU availability query failed')
        available=[tuple(map(int,row.split(','))) for row in data.decode().splitlines()]
        choices=[r for r in available if r[1]>=30000]
        if choices:
            gpu=max(choices,key=lambda r:r[1])[0];break
        await asyncio.sleep(60)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),FDANYONE_WAYMO_ROOT=str(ROOT),
        FDANYONE_RIFE_SOURCE=str(ROOT/'inputs_native/frame_mapping.json'),FDANYONE_CONDITIONING_VIEWS='6',
        FDANYONE_RETAINED_VIEWS='6',FDANYONE_MAX_RADIUS_RATIO='1.75',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TORCH_CUDNN_V8_API_DISABLED='1',CUDNN_FRONTEND_DISABLE='1',NVIDIA_TF32_OVERRIDE='0')
    env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    state.update(state='running',physical_gpu=gpu);save(state)
    async def run(stage,python,script,*args):
        state.update(stage=stage);save(state)
        with (ROOT/'logs'/f'{stage}.log').open('x') as log:
            child=await asyncio.create_subprocess_exec(python,str(S/'scripts'/script),*args,env=env,cwd=S/'repo',
                stdout=log,stderr=asyncio.subprocess.STDOUT,pass_fds=(fd,))
            state['child_pid']=child.pid;save(state);code=await child.wait()
        (ROOT/'logs'/f'{stage}.exit').write_text(str(code)+'\n')
        if code:raise RuntimeError(f'{stage} exited {code}; see logs/{stage}.log')
        state['done'].append(stage);state.pop('child_pid',None);save(state)
    try:
        await run('rife',PY,'interpolate.py')
        await run('frames',PY,'experiment_prepare.py','frames')
        await run('segment',SAM,'segment_target.py')
        await run('composite',PY,'prepare_foreground.py')
        await run('input_check',PY,'validate_short.py','input')
        await run('mask_gate',PY,'batch_gate.py','mask')
        await run('preflight',PY,'preflight.py')
        await run('fit6',PY,'fit_conditioning.py')
        await run('crop_audit',PY,'crop_audit.py')
        await run('final_gate',PY,'batch_gate.py','final')
        await run('generate',PY,'generate.py','smoke')
        await run('video_check',PY,'validate_short.py','output')
        state.update(state='generated_technical_checks_passed_visual_review_pending')
    except Exception as error:
        state.update(state='failed',error=f'{type(error).__name__}: {error}');save(state);raise
    finally:save(state)


if __name__=='__main__':
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        asyncio.run(main(lock.fileno()))
