"""Bounded asynchronous continuation from two generated pilots to animated assets."""
import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

R=Path(__file__).resolve().parents[1];W=R.parent
FTGS='/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python'
SAM='/mnt/ojc/miniconda3/envs/4danyone-sam2/bin/python'


async def case(name,gpu):
    mode=os.environ.get('FDANYONE_ASSET_MODE','smoke');assert mode in ('smoke','turbo')
    views=6 if mode=='smoke' else 24
    label=name+('_six' if mode=='smoke' else '_twentyfour');source=W/'experiments'/name/'outputs'/mode
    path=R/'research'/f'{label}-job.json'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),FDANYONE_ASSET_NAME=label,
        FDANYONE_ASSET_SOURCE=str(source),FDANYONE_ASSET_GPU_IDS=str(gpu) if views==6 else '4,5,6,7',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    env['CUDA_HOME']='/mnt/ojc/miniconda3/envs/asset_harvester_2026'
    env['PATH']=env['CUDA_HOME']+'/bin:'+env['PATH']
    env['TORCH_EXTENSIONS_DIR']=str(R/'deps/torch_extensions')
    env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    state={'label':label,'source_experiment':name,'physical_gpu':gpu,'mask_gpu_ids':[int(x) for x in env['FDANYONE_ASSET_GPU_IDS'].split(',')],'views':views,'state':'waiting_for_generation',
        'scope':'Generated-view pilot baseline; generation and reconstruction quality remain separate acceptance gates'}
    def save(stage):
        state['state']=stage;state['updated_at']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(path)
        summary_path=W/'status.json'
        summary=json.loads(summary_path.read_text())
        summary.setdefault('asset_jobs',{})[label]=dict(state)
        summary['updated_at']=state['updated_at']
        temporary=summary_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(summary,indent=2)+'\n');temporary.replace(summary_path)
    async def run(stage,python,script,*args):
        save(stage)
        log=R/'logs'/f'{label}-{stage}.log'
        if log.exists():raise FileExistsError(log)
        with log.open('w') as out:
            process=await asyncio.create_subprocess_exec(python,str(script),*map(str,args),env=env,
                cwd=W/'bridge/backend',stdout=out,stderr=asyncio.subprocess.STDOUT)
            code=await process.wait()
        log.with_suffix('.exit').write_text(str(code)+'\n')
        if code:raise RuntimeError(f'{stage} exit {code}; inspect {log}')
    save('waiting_for_generation')
    try:
        deadline=asyncio.get_running_loop().time()+2400
        while True:
            p=W/'experiments'/name/'experiment_status.json'
            status=json.loads(p.read_text())
            if status['stage']==mode and status['state']=='generated_visual_review_pending':break
            if status['state']=='failed':raise RuntimeError('Generation failed; no reconstruction started')
            if asyncio.get_running_loop().time()>deadline:raise TimeoutError('Generation wait exceeded 40 minutes')
            await asyncio.sleep(20)
        validation=json.loads((source/'preview/final_validation.json').read_text())
        assert (source/'videos/05.mp4').is_file() and validation
        if shutil.disk_usage(R).free<30*1024**3:raise RuntimeError('Less than 30 GiB disk reserve')
        proc=await asyncio.create_subprocess_exec('nvidia-smi',f"--id={env['FDANYONE_ASSET_GPU_IDS']}",'--query-gpu=memory.free','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE)
        output,_=await proc.communicate()
        if proc.returncode or min(map(int,output.splitlines()))<20000:raise RuntimeError('Need 20 GiB free on assigned shared GPUs; do not terminate other processes')
        await run('masks',SAM,R/'scripts/full_masks.py')
        await run('export',FTGS,R/'scripts/export_sequence.py')
        data=R/'data'/label;out=R/'outputs'/label
        if out.exists():raise FileExistsError(out)
        await run('train',FTGS,W/'bridge/backend/src/simple_trainer_freetime_4d_pure_relocation.py','default_keyframe',
            '--data-dir',data,'--init-npz-path',data/'init.npz','--result-dir',out,'--start-frame',0,'--end-frame',121,
            '--data-factor',1,'--test-every',6 if views==6 else 8,'--batch-size',1,'--init-scale',1.0,
            '--alpha-mode','transparent','--random-bkgd','--lambda-alpha',.1,
            '--max-steps',5000,'--save-steps',1000,3000,5000,'--eval-steps',5000,'--eval-sample-every',30,
            '--render-traj-n-frames',121,'--render-traj-time-frames',121,'--render-traj-fps',25,
            '--export-ply','--export-ply-format','ftgs','--disable-viewer')
        await run('validate',FTGS,R/'scripts/validate_asset.py')
        save('asset_validated_visual_review_pending')
    except Exception as error:
        state['error']=f'{type(error).__name__}: {error}';save('failed')
    return state


async def main():
    jobs=[('pilot_005',6),('pilot_043',7)] if len(sys.argv)==1 else [(part.split(':')[0],int(part.split(':')[1])) for part in sys.argv[1:]]
    assert 1<=len(jobs)<=4 and len(set(gpu for _,gpu in jobs))==len(jobs)
    if os.environ.get('FDANYONE_ASSET_MODE')=='turbo':assert len(jobs)==1
    assert all(name.startswith('pilot_') and name.removeprefix('pilot_').isdigit() and gpu in range(4,8) for name,gpu in jobs)
    results=await asyncio.gather(*map(lambda job:case(*job),jobs))
    label='-'.join(name for name,_ in jobs)
    (R/'research'/f'{label}-assets.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results),flush=True)
    if any(x['state']=='failed' for x in results):raise SystemExit(1)


if __name__=='__main__':asyncio.run(main())
