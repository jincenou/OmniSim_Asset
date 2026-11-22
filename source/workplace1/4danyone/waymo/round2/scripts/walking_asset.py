"""Reconstruct reviewed24-view walking data on GPU6/7 alongside Q1/Q2 on4/5."""
import asyncio
import fcntl
import os
from pathlib import Path
import shutil
from after_batch import ROUND,W,P,PY,read,write,now
from resource_lease import lease,wait_memory

R=ROUND/'reconstruction';ROOT=ROUND/'experiments/walk063_rootlocal24_v1';SOURCE=ROOT/'outputs/turbo'
STATE=ROUND/'walking_asset_run.json';LABEL='walk063_rootlocal24'
SAM='/mnt/ojc/miniconda3/envs/4danyone-sam2/bin/python'
FTGS='/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python'


async def main():
    with (ROUND/'walking_asset.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if STATE.exists():raise FileExistsError(STATE)
        state={'state':'waiting_for_24view_visual_gate','pid':os.getpid(),'started_at':now(),'stages':[]}
        def save(**fields):state.update(fields,updated_at=now());write(STATE,state)
        save()
        try:
            while True:
                generation=read(ROUND/'walk24_run.json')
                if generation['state']=='failed':raise RuntimeError('24-view generation failed; no reconstruction launched')
                review=ROUND/'reviews/walk063_24.json'
                if generation['state']=='24_view_generated_visual_review_pending' and review.exists():
                    if not read(review)['allow_reconstruction']:raise RuntimeError('Visual gate rejected this generated sequence')
                    break
                await asyncio.sleep(60)
            async with lease([6,7]) as fds:
                await wait_memory([6,7],16000)
                if shutil.disk_usage(ROUND).free<30*1024**3:raise RuntimeError('Disk reserve below30GiB')
                if (R/'data'/LABEL).exists() or (R/'outputs'/LABEL).exists():raise FileExistsError('Preserve previous walking asset attempt before retry')
                # Same source input and timeline; preserve actual upstream mapping.
                mapping=ROOT/'inputs/frame_mapping.json'
                if not mapping.exists():mapping.symlink_to(W/'experiments/pilot_063/inputs/frame_mapping.json')
                env=os.environ.copy();env.update(FDANYONE_ASSET_NAME=LABEL,FDANYONE_ASSET_SOURCE=str(SOURCE),
                    FDANYONE_ASSET_GPU_IDS='6,7',CUDA_VISIBLE_DEVICES='6,7',CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    CUDA_HOME='/mnt/ojc/miniconda3/envs/asset_harvester_2026',TORCH_EXTENSIONS_DIR=str(W/'reconstruction/deps/torch_extensions'),
                    MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.9')
                env['PATH']=env['CUDA_HOME']+'/bin:'+env['PATH'];env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
                async def stage(name,command):
                    log=ROUND/'logs'/f'walking_asset_{name}.log'
                    with log.open('xb') as handle:
                        process=await asyncio.create_subprocess_exec(*command,cwd=ROUND/'backend',env=env,
                            stdout=handle,stderr=asyncio.subprocess.STDOUT,pass_fds=(*fds,own.fileno()))
                        record={'stage':name,'pid':process.pid,'command':command,'log':str(log),'started_at':now()}
                        state['stages'].append(record);save(state='running',stage=name)
                        code=await process.wait();log.with_suffix('.exit').write_text(str(code)+'\n')
                    record.update(exit_code=code,ended_at=now());save()
                    if code:raise RuntimeError(f'{name} exited{code}; see{log}')
                await stage('masks',[SAM,str(R/'scripts/full_masks.py')])
                env['CUDA_VISIBLE_DEVICES']=''
                await stage('export',[FTGS,str(R/'scripts/export_sequence.py')])
                env['CUDA_VISIBLE_DEVICES']='6'
                await stage('train',[FTGS,str(ROUND/'scripts/train_walking.py')])
                await stage('validate',[FTGS,str(R/'scripts/validate_asset.py')])
                out=R/'outputs'/LABEL
                await stage('metrics',[FTGS,str(ROUND/'scripts/evaluate_asset.py'),'--checkpoint',str(out/'ckpts/ckpt_4999.pt'),
                    '--data',str(R/'data'/LABEL),'--output',str(out/'quality_metrics')])
                save(state='walking_asset_technical_complete_trajectory_replay_pending',asset=str(out/'ckpt_4999.ftgs.ply'),
                    preview=str(out/'preview/fixed_views.mp4'),visual_review='pending')
        except Exception as error:save(state='failed',error=f'{type(error).__name__}: {error}');raise


if __name__=='__main__':asyncio.run(main())
