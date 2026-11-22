"""Reviewed 24-view demos to 5k dynamic assets, with persistent stage records."""
import asyncio
import fcntl
import os
from pathlib import Path
import shutil
from after_batch import ROUND,W,read,write,now
from resource_lease import lease,wait_memory
PY='/mnt/ojc/miniconda3/envs/4danyone/bin/python'
SAM='/mnt/ojc/miniconda3/envs/4danyone-sam2/bin/python'
FTGS='/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python'
R=ROUND/'reconstruction'

async def main():
    with (ROUND/'new_assets.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=ROUND/'new_assets_queue.json'
        if path.exists():raise FileExistsError(path)
        state={'state':'waiting_for_generated_visual_reviews','pid':os.getpid(),'started_at':now(),'jobs':{}}
        def save():state['updated_at']=now();write(path,state)
        save()
        async def case(job):
            name=job['id'];label=job['asset_label'];root=Path(job['output']);source=root/'outputs/turbo'
            gpu=job['gpu'];out=R/'outputs'/label;data=R/'data'/label
            row={'state':'waiting_for_generated_visual_review','gpu':gpu,'label':label,'stages':[]};state['jobs'][name]=row;save()
            try:
                while True:
                    upstream=read(ROUND/'new24_queue.json');generated=upstream['jobs'].get(name,{})
                    if generated.get('state')=='failed':raise RuntimeError('Upstream generation failed; no reconstruction')
                    review=ROUND/'reviews'/(label+'_24.json')
                    if generated.get('state')=='24view_complete_visual_review_pending' and review.exists():
                        gate=read(review)
                        if not gate['allow_reconstruction']:
                            row.update(state='rejected_by_visual_review',review=str(review));save();return
                        if gate['source']!=str(source):raise ValueError('Review refers to a different generation')
                        break
                    await asyncio.sleep(30)
                if out.exists() or data.exists():raise FileExistsError('Previous output exists; explicit resume required')
                if shutil.disk_usage(ROUND).free<40*1024**3:raise RuntimeError('Disk reserve below40GiB')
                env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',FDANYONE_ASSET_NAME=label,
                    FDANYONE_ASSET_SOURCE=str(source),CUDA_HOME='/mnt/ojc/miniconda3/envs/asset_harvester_2026',
                    TORCH_EXTENSIONS_DIR=str(W/'reconstruction/deps/torch_extensions'),MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='8.9',
                    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
                env['PATH']=env['CUDA_HOME']+'/bin:'+env['PATH'];env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
                async def stage(stage_name,python,script,*args,handles=()):
                    command=[python,str(script),*map(str,args)];log=ROUND/'logs'/f'new_asset_{label}_{stage_name}.log'
                    with log.open('xb') as stream:
                        child=await asyncio.create_subprocess_exec(*command,cwd=ROUND/'backend',env=env,stdout=stream,
                            stderr=asyncio.subprocess.STDOUT,pass_fds=(*handles,own.fileno()))
                        item={'stage':stage_name,'pid':child.pid,'command':command,'log':str(log),'state':'running','started_at':now()}
                        row['stages'].append(item);row.update(state='running',stage=stage_name);save();code=await child.wait()
                    item.update(exit_code=code,state='complete' if code==0 else 'failed',ended_at=now());log.with_suffix('.exit').write_text(str(code)+'\n');save()
                    if code:raise RuntimeError(f'{stage_name} exit{code}; preserved {log}')
                row.update(state='waiting_for_mask_resources');save()
                async with lease([4,5,6,7]) as handles:
                    await wait_memory([4,5,6,7],16000)
                    env.update(CUDA_VISIBLE_DEVICES='4,5,6,7',FDANYONE_ASSET_GPU_IDS='4,5,6,7')
                    await stage('masks',SAM,R/'scripts/full_masks.py',handles=handles)
                env['CUDA_VISIBLE_DEVICES']=''
                await stage('export',FTGS,R/'scripts/export_sequence.py')
                row.update(state='waiting_for_training_resources');save()
                async with lease([gpu]) as handles:
                    await wait_memory([gpu],6000);env['CUDA_VISIBLE_DEVICES']=str(gpu)
                    await stage('train5k',FTGS,ROUND/'scripts/train_new_asset.py',handles=handles)
                    await stage('validate',FTGS,R/'scripts/validate_asset.py',handles=handles)
                    await stage('metrics',FTGS,ROUND/'scripts/evaluate_asset.py','--checkpoint',out/'ckpts/ckpt_4999.pt',
                        '--data',data,'--output',out/'quality_metrics',handles=handles)
                    original=Path(job['source']);demo_job=ROUND/'demo_jobs'/f'{label}.json'
                    write(demo_job,{'label':label,'source':str(source),'input':str(root/'inputs/main.mp4'),
                        'mapping':str(root/'inputs/frame_mapping.json'),'native_mapping':str(original/'inputs_native/frame_mapping.json'),
                        'data':str(data),'checkpoint':str(out/'ckpts/ckpt_4999.pt'),'output':str(ROUND/'demos'/label)})
                    await stage('demo',FTGS,ROUND/'scripts/render_demo.py','--job',demo_job,handles=handles)
                if read(out/'preview/validation.json')['status']!='trained_rendered_exported_validated':raise ValueError('Missing asset validation')
                row.update(state='dynamic_asset_and_demo_complete_visual_review_pending',asset=str(out/'ckpt_4999.ftgs.ply'),
                    demo=str(ROUND/'demos'/label/'four_panel.mp4'),production_approved=False)
            except Exception as error:row.update(state='failed',error=f'{type(error).__name__}: {error}')
            finally:save()
        await asyncio.gather(*map(case,read(ROUND/'next24_plan.json')['jobs']))
        failures=[k for k,v in state['jobs'].items() if v['state']=='failed']
        state.update(state='completed_with_failures' if failures else 'all_reviewed_jobs_processed',failures=failures);save()
        if failures:raise SystemExit(1)

if __name__=='__main__':asyncio.run(main())
