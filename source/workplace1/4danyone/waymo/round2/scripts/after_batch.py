"""Start one isolated walking diagnostic after the baseline twelve have finished."""
import asyncio
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

ROUND=Path(__file__).resolve().parents[1]
W=ROUND.parent; P=W.parent; B=W/'batch100'
ROOT=ROUND/'experiments/walk063_rootlocal_v1'
PY='/mnt/ojc/miniconda3/envs/4danyone/bin/python'
STATE=ROUND/'next_run.json'


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def read(path):return json.loads(path.read_text())
def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def gate(status,controller,control):
    if control['target_distinct_assets']!=12:
        raise RuntimeError('Baseline target changed; reassess this explicit twelve-asset handoff')
    done=status['completed_distinct_technical_assets']
    if done>12:raise RuntimeError('Baseline exceeded twelve; do not hide the overshoot')
    if controller['state']=='candidate_pool_exhausted' and done<12:
        raise RuntimeError('Candidate pool exhausted before twelve assets')
    return done==12 and not status['active'] and controller['state']=='target_reached'


def verify_assets(status):
    seeds=status['seed_assets']
    records=[read(B/'jobs'/f'{name}.json') for name,info in status['jobs'].items() if info['state']=='complete']
    outputs=[W/'reconstruction/outputs'/seed['label'] for seed in seeds]
    outputs.extend(Path(record['asset']).parent for record in records)
    if len(set(outputs))!=12:raise RuntimeError('Expected twelve distinct output directories')
    def validate(out):
        if read(out/'preview/validation.json')['status']!='trained_rendered_exported_validated':
            raise RuntimeError(f'Unvalidated asset: {out}')
        required=('ckpt_4999.ftgs.ply','ckpts/ckpt_4999.pt','asset.json','preview/fixed_views.mp4')
        if not all((out/name).is_file() and (out/name).stat().st_size>0 for name in required):
            raise RuntimeError(f'Missing artifact: {out}')
        return str(out)
    return list(map(validate,outputs))


class Runner:
    def __init__(self,lock_fd):
        self.fds=[lock_fd]
        self.state={'pid':os.getpid(),'started_at':now(),'state':'waiting_for_baseline_12',
                    'gpu_ids_allowed':[4,5,6,7],'output':str(ROOT),'stage_history':[],
                    'scope':'root-local six-view diagnostic; visual review required before scaling'}

    def update(self,**fields):
        self.state.update(fields,updated_at=now());write(STATE,self.state)

    async def stage(self,name,command,env):
        path=ROUND/'logs'/f'walk063_{name}.log'
        if path.exists():raise FileExistsError(path)
        with path.open('xb') as log:
            process=await asyncio.create_subprocess_exec(*command,cwd=P/'repo',env=env,
                stdout=log,stderr=asyncio.subprocess.STDOUT,pass_fds=tuple(self.fds))
            record={'stage':name,'pid':process.pid,'command':command,'log':str(path),'started_at':now()}
            self.state['stage_history'].append(record)
            self.update(state='running',stage=name,child_pid=process.pid)
            code=await process.wait()
            path.with_suffix('.exit').write_text(str(code)+'\n')
            record.update(exit_code=code,ended_at=now());self.update(child_pid=None)
            if code:raise RuntimeError(f'{name} exited {code}; see {path}')

    async def main(self):
        self.update()
        baseline_lock=(B/'controller.lock').open('a')
        while True:
            status=read(B/'status.json')
            if gate(status,read(B/'controller.json'),read(B/'control.json')):
                try:fcntl.flock(baseline_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:pass
                else:break
            self.update(baseline_completed=status['completed_distinct_technical_assets'],baseline_active=status['active'])
            await asyncio.sleep(60)
        self.fds.append(baseline_lock.fileno())
        if not gate(read(B/'status.json'),read(B/'controller.json'),read(B/'control.json')):
            raise RuntimeError('Baseline state changed during lock acquisition')
        completed=verify_assets(read(B/'status.json'))
        self.update(state='waiting_for_gpu',baseline_completed=12,baseline_active=[],verified_baseline_outputs=completed)
        while True:
            process=await asyncio.create_subprocess_exec('nvidia-smi','--id=4,5,6,7',
                '--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE)
            data,_=await process.communicate()
            if process.returncode:raise RuntimeError('GPU query failed')
            rows=[line.split(',') for line in data.decode().strip().splitlines()]
            candidates=[(int(free),int(index),uuid.strip()) for index,uuid,free in rows if int(index) in (4,5,6,7) and int(free)>=47000]
            if candidates:
                free,gpu,uuid=max(candidates);break
            self.update(state='waiting_for_gpu',required_free_mib=47000)
            await asyncio.sleep(60)
        if shutil.disk_usage(P).free<30*1024**3:raise RuntimeError('Less than 30 GiB free')
        if ROOT.exists():raise FileExistsError(ROOT)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',
            FDANYONE_WAYMO_ROOT=str(ROOT),FDANYONE_RETAINED_VIEWS='6')
        env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        self.update(physical_gpu=gpu,gpu_uuid=uuid,free_mib_before_start=free)
        await self.stage('conditioning',[PY,str(ROUND/'scripts/walk_conditioning.py')],env)
        await self.stage('generate',[PY,str(W/'scripts/generate.py'),'smoke'],env)
        out=ROOT/'outputs/smoke'
        # Official writer retains cameras but omits custom root-trajectory fields.
        # Package them explicitly so the generated data cannot lose its frame mapping.
        shutil.copy2(ROOT/'preflight/root_motion.npz',out/'root_motion.npz')
        shutil.copy2(ROOT/'research/root_local.json',out/'root_local.json')
        payload=read(out/'cameras.json')
        if payload['world_frame']['name']!='root_local_human_world':raise RuntimeError('World frame lost by writer')
        payload['root_motion']={'file':'root_motion.npz','sha256':sha(out/'root_motion.npz'),
            'local_to_canonical_key':'local_to_canonical','time_key':'frame_times',
            'motion_world_to_local_key':'motion_to_local','provenance':'root_local.json'}
        write(out/'cameras.json',payload)
        await self.stage('preview',[PY,str(P/'scripts/visualize.py'),str(out)],env)
        await self.stage('validate',[PY,str(W/'scripts/validate_output.py'),'smoke'],env)
        self.update(state='six_view_complete_visual_review_pending',stage='complete',
                    preview=str(out/'preview/multiview.mp4'),validation=str(out/'preview/final_validation.json'))
        baseline_lock.close()


async def main():
    (ROUND/'logs').mkdir(exist_ok=True)
    with (ROUND/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if STATE.exists():raise FileExistsError(f'Preserve prior run and choose a new experiment before retry: {STATE}')
        runner=Runner(lock.fileno())
        try:await runner.main()
        except Exception as error:
            runner.update(state='failed',error=f'{type(error).__name__}: {error}')
            raise


if __name__=='__main__':asyncio.run(main())
