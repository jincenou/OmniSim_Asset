"""Persistent 24-view batch: bounded async jobs, shared GPU leases, stage resume."""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
from zoneinfo import ZoneInfo

R=Path(__file__).resolve().parents[1];W=R.parent;B=W/'batch100'
PY='/mnt/ojc/miniconda3/envs/4danyone/bin/python'
SAM='/mnt/ojc/miniconda3/envs/4danyone-sam2/bin/python'
FTGS='/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python'


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


class Pool:
    def __init__(self):
        self.free={4,5,6,7};self.waiters=[];self.condition=asyncio.Condition()

    @asynccontextmanager
    async def lease(self,count,minimum_mib):
        ticket=object()
        async with self.condition:
            self.waiters.append(ticket)
            await self.condition.wait_for(lambda:self.waiters[0] is ticket and len(self.free)>=count)
            ids=sorted(self.free)[:count];self.free.difference_update(ids);self.waiters.pop(0);self.condition.notify_all()
        try:
            while True:
                p=await asyncio.create_subprocess_exec('nvidia-smi','--id='+','.join(map(str,ids)),
                    '--query-gpu=memory.free','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE)
                output,_=await p.communicate()
                if p.returncode:raise RuntimeError('GPU query failed')
                if min(map(int,output.splitlines()))>=minimum_mib:break
                write(B/'resource_wait.json',{'updated_at':now(),'gpus':ids,'required_free_mib':minimum_mib,'action':'wait; leave other processes running'})
                await asyncio.sleep(60)
            yield ids
        finally:
            async with self.condition:
                self.free.update(ids);self.condition.notify_all()


class Batch:
    def __init__(self,fd):
        self.fd=fd;self.plan=json.loads((B/'plan.json').read_text());self.pool=Pool();self.condition=asyncio.Condition()
        self.records={p.stem:json.loads(p.read_text()) for p in (B/'jobs').glob('*.json')}
        self.active=set();self.cursor=0
        for seed in self.plan['seed_assets']:
            assert json.loads((R/'outputs'/seed['label']/'preview/validation.json').read_text())['status']=='trained_rendered_exported_validated'

    def count(self):return len(self.plan['seed_assets'])+sum(x['state']=='complete' for x in self.records.values())

    def target(self):
        control=B/'control.json'
        value=json.loads(control.read_text())['target_distinct_assets'] if control.exists() else self.plan['target_distinct_assets']
        assert isinstance(value,int) and value>=len(self.plan['seed_assets'])
        return value

    def summary(self):
        value={'updated_at':now(),'target':self.target(),'completed_distinct_technical_assets':self.count(),
            'seed_assets':self.plan['seed_assets'],'active':sorted(self.active),'attempted':len(self.records),
            'failed_or_rejected':sum(x['state']=='failed' for x in self.records.values()),
            'candidate_reserve':len(self.plan['candidates']),'views_per_asset':24,'frames':121,'gpu_ids':[4,5,6,7],
            'quality_status':'Technical completion is not manual appearance approval','jobs':{k:{x:v[x] for x in ('state','stage','updated_at') if x in v} for k,v in self.records.items()}}
        write(B/'status.json',value)

    def save(self,record):
        record['updated_at']=now();self.records[record['name']]=record;write(B/'jobs'/f"{record['name']}.json",record);self.summary()

    async def next_job(self):
        async with self.condition:
            while True:
                if self.count()>=self.target():return None
                if self.count()+len(self.active)>=self.target():
                    await self.condition.wait();continue
                while self.cursor<len(self.plan['candidates']):
                    index=self.cursor;self.cursor+=1;name=f'batch_{index:04d}'
                    old=self.records.get(name)
                    if old and old['state'] in ('complete','failed'):continue
                    self.active.add(name)
                    record=old or {'name':name,'candidate_index':index,'identity_key':self.plan['candidates'][index]['identity_key'],
                        'state':'running','done':[],'started_at':now(),'stage':'queued','attempts':{}}
                    record['state']='running';self.save(record);return record
                return None

    async def run(self,record,stage,python,script,*args,env):
        if stage in record['done']:return
        while shutil.disk_usage(R).free<self.plan['disk_reserve_gib']*1024**3:
            record['stage']='waiting_for_disk';self.save(record);await asyncio.sleep(120)
        record['stage']=stage;attempt=record['attempts'].get(stage,0)+1;record['attempts'][stage]=attempt;self.save(record)
        path=B/'logs'/f"{record['name']}-{stage}-{attempt}.log"
        command=[python,str(script),*map(str,args)]
        with path.open('x') as log:
            process=await asyncio.create_subprocess_exec(*command,env=env,cwd=W.parent/'repo',stdout=log,
                stderr=asyncio.subprocess.STDOUT,pass_fds=(self.fd,))
            record['pid']=process.pid;record['log']=str(path);self.save(record)
            code=await process.wait()
        path.with_suffix('.exit').write_text(str(code)+'\n')
        if code:raise RuntimeError(f'{stage} exit {code}; {path}')
        record['done'].append(stage);record.pop('pid',None);self.save(record)

    def archive(self,path):
        if path.exists():path.rename(path.with_name(path.name+f'_interrupted_{time.time_ns()}'))

    def environment(self,root,ids):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,ids)),FDANYONE_WAYMO_ROOT=str(root),
            FDANYONE_RIFE_SOURCE=str(root/'inputs_native/frame_mapping.json'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            TORCH_CUDNN_V8_API_DISABLED='1',CUDNN_FRONTEND_DISABLE='1',NVIDIA_TF32_OVERRIDE='0',
            FDANYONE_CONDITIONING_VIEWS='24',FDANYONE_RETAINED_VIEWS='24',FDANYONE_MAX_RADIUS_RATIO='1.75',
            CUDA_HOME='/mnt/ojc/miniconda3/envs/asset_harvester_2026',TORCH_EXTENSIONS_DIR=str(R/'deps/torch_extensions'))
        env['PATH']=env['CUDA_HOME']+'/bin:'+env['PATH'];env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        return env

    async def case(self,record):
        name=record['name'];root=W/'experiments'/name;row=self.plan['candidates'][record['candidate_index']]
        label=name+'_twentyfour';data=R/'data'/label;out=R/'outputs'/label
        try:
            if 'prepare' not in record['done']:
                self.archive(root)
                for folder in ['inputs','inputs_native','research','segmentation/frames','logs','previews']:(root/folder).mkdir(parents=True,exist_ok=True)
                write(root/'candidate.json',row)
                async with self.pool.lease(1,16000) as ids:
                    write(root/'config.json',{'name':name,'candidate':name,'candidate_record':str(root/'candidate.json'),
                        'source_split':'val','start_frame':row['observations'][0]['frame_index'],'gpu':ids[0],'portrait_crop':True,
                        'positive_points_box_fraction':[[.5,.25],[.5,.5],[.45,.75]],
                        'prompt_rationale':'Native tracked identity; automated screening, appearance review remains pending'})
                    env=self.environment(root,ids)
                    # Interrupted preparation restarts in an archived isolated directory.
                    prep=['native','rife','frames','segment','composite','input_check','mask_gate','preflight','fit24','crop_audit','final_gate','contact']
                    record['done']=[s for s in record['done'] if s not in prep]
                    await self.run(record,'native',PY,W/'scripts/experiment_prepare.py','native',env=env)
                    await self.run(record,'rife',PY,W/'scripts/interpolate.py',env=env)
                    await self.run(record,'frames',PY,W/'scripts/experiment_prepare.py','frames',env=env)
                    await self.run(record,'segment',SAM,W/'scripts/segment_target.py',env=env)
                    await self.run(record,'composite',PY,W/'scripts/prepare_foreground.py',env=env)
                    await self.run(record,'input_check',PY,W/'scripts/validate_input.py',env=env)
                    await self.run(record,'mask_gate',PY,W/'scripts/batch_gate.py','mask',env=env)
                    await self.run(record,'preflight',PY,W/'scripts/preflight.py',env=env)
                    await self.run(record,'fit24',PY,W/'scripts/fit_conditioning.py',env=env)
                    await self.run(record,'crop_audit',PY,W/'scripts/crop_audit.py',env=env)
                    await self.run(record,'final_gate',PY,W/'scripts/batch_gate.py','final',env=env)
                    await self.run(record,'contact',PY,W/'scripts/experiment_review.py',env=env)
                    record['done'].append('prepare');self.save(record)
            if 'generate' not in record['done']:
                self.archive(root/'outputs/turbo')
                async with self.pool.lease(4,47000) as ids:
                    await self.run(record,'generate',PY,W/'scripts/generate.py','turbo',env=self.environment(root,ids))
            cpu_env=self.environment(root,[4,5,6,7])
            await self.run(record,'video_preview',PY,W.parent/'scripts/visualize.py',root/'outputs/turbo',env=cpu_env)
            await self.run(record,'video_check',PY,W/'scripts/validate_output.py','turbo',env=cpu_env)
            def asset_env(ids):
                env=self.environment(root,ids);env.update(FDANYONE_ASSET_NAME=label,FDANYONE_ASSET_SOURCE=str(root/'outputs/turbo'),FDANYONE_ASSET_GPU_IDS=','.join(map(str,ids)))
                return env
            if 'masks' not in record['done']:
                self.archive(data)
                async with self.pool.lease(4,16000) as ids:
                    await self.run(record,'masks',SAM,R/'scripts/full_masks.py',env=asset_env(ids))
            if 'export' not in record['done']:
                self.archive(data/'init.npz');self.archive(data/'sparse')
                await self.run(record,'export',FTGS,R/'scripts/export_sequence.py',env=asset_env([7]))
            if 'train' not in record['done']:
                checkpoints=sorted((out/'ckpts').glob('ckpt_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
                complete=(out/'ckpts/ckpt_4999.pt').exists() and (out/'ckpt_4999.ftgs.ply').exists() and (out/'videos/traj_4d_step4999.mp4').exists()
                if complete:record['done'].append('train');self.save(record)
                else:
                    resumable=[p for p in checkpoints if int(p.stem.split('_')[-1])<4999]
                    extra=['--ckpt-path',resumable[-1],'--resume-mode','resume'] if resumable else []
                    async with self.pool.lease(1,4000) as ids:
                        await self.run(record,'train',FTGS,W/'bridge/backend/src/simple_trainer_freetime_4d_pure_relocation.py','default_keyframe',
                            '--data-dir',data,'--init-npz-path',data/'init.npz','--result-dir',out,'--start-frame',0,'--end-frame',121,
                            '--data-factor',1,'--test-every',8,'--batch-size',1,'--init-scale',1.0,'--alpha-mode','transparent',
                            '--random-bkgd','--lambda-alpha',.1,'--max-steps',5000,'--save-steps',1000,3000,5000,
                            '--eval-steps',5000,'--eval-sample-every',30,'--render-traj-n-frames',121,'--render-traj-time-frames',121,
                            '--render-traj-fps',25,'--export-ply','--export-ply-format','ftgs','--disable-viewer',*extra,env=asset_env(ids))
            if 'validate' not in record['done']:
                async with self.pool.lease(1,4000) as ids:
                    await self.run(record,'validate',FTGS,R/'scripts/validate_asset.py',env=asset_env(ids))
            validation=json.loads((out/'preview/validation.json').read_text())
            assert validation['status']=='trained_rendered_exported_validated' and validation['observation_views']==24
            assert (out/'asset.json').exists()
            record.update(state='complete',stage='complete',asset=str(out/'ckpt_4999.ftgs.ply'),preview=str(out/'preview/fixed_views.mp4'),
                quality='automatic_checks_passed_manual_appearance_review_pending')
        except Exception as error:
            record.update(state='failed',error=f'{type(error).__name__}: {error}')
        finally:
            self.save(record)
            async with self.condition:
                self.active.discard(name);self.condition.notify_all();self.summary()

    async def worker(self):
        while (record:=await self.next_job()) is not None:await self.case(record)

    async def main(self):
        write(B/'controller.json',{'pid':os.getpid(),'started_at':now(),'state':'running','script':__file__})
        await asyncio.gather(*map(lambda _:self.worker(),range(self.plan['max_inflight'])))
        self.summary()
        write(B/'controller.json',{'pid':os.getpid(),'updated_at':now(),'state':'target_reached' if self.count()>=self.target() else 'candidate_pool_exhausted',
            'completed_distinct_technical_assets':self.count(),'target':self.target()})


if __name__=='__main__':
    with (B/'controller.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        asyncio.run(Batch(lock.fileno()).main())
