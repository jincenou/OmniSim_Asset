"""Bounded asynchronous people queue; pose approval is an explicit per-person gate."""
import argparse,asyncio,json,os,sys,time
from pathlib import Path
from prepare_person import ROOT,selected,sha

CPU=asyncio.Semaphore(2)
GPU=asyncio.Semaphore(4)
STATE={}

def save():
    path=ROOT/'queue_status.json';temporary=path.with_suffix('.tmp')
    STATE['updated_at']=time.time();temporary.write_text(json.dumps(STATE,indent=2,allow_nan=False)+'\n');temporary.replace(path)

async def stage(ident,name,gpu,minimum=6000):
    row=STATE['people'][ident]
    for key in ('pid','exit_code','finished','started','log'):row.pop(key,None)
    row.update(stage=name,state='queued',updated_at=time.time());save()
    record=ROOT/'logs'/f'{ident}_{name}_complete.json'
    if record.exists():
        previous=json.loads(record.read_text())
        assert previous['selection_sha256']==sha(ROOT/'SELECTED_TRACKS.json')
        assert previous['baseline_sha256']==sha(ROOT/'BASELINE_LOCK.json')
        def current_path(path):
            original=ROOT/path
            old=ROOT/'backend/runs'/f'{ident}_A'
            if original.is_relative_to(old) and (ROOT/'assets'/ident).is_dir():
                return ROOT/'assets'/ident/original.relative_to(old)
            return original
        assert all(current_path(path).is_file() and sha(current_path(path))==value for path,value in previous['outputs'].items())
        row.update(state='verified_stage_reused',stage=name);save();return
    async with GPU if gpu else CPU:
        command=[sys.executable,'-u',str(ROOT/'scripts/execute_stage.py'),name,ident]
        if gpu:
            command=[sys.executable,'-u',str(ROOT/'scripts/run_leased.py'),'--name',f'{ident}_{name}',
                '--preferred-gpu',str(4+(int(ident[1:])-1)%4),'--minimum-free-mib',str(minimum),*command]
        log=ROOT/'logs'/f'{ident}_{name}_controller.log'
        with log.open('a') as stream:
            child=await asyncio.create_subprocess_exec(*command,cwd=ROOT,stdout=stream,stderr=stream,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            row.update(state='running_or_waiting_shared_gpu',pid=child.pid,log=str(log),started=time.time());save()
            code=await child.wait()
        row.update(exit_code=code,finished=time.time());save()
        if code:raise RuntimeError(f'{name} exited {code}; see {log}')
    outputs={
        'prepare':[f'inputs/{ident}/candidate.json',f'inputs/{ident}/crop_unsegmented.mp4'],
        'segment':[f'inputs/{ident}/research/batch_gate_mask.json',f'inputs/{ident}/source_crop.mp4'],
        'motion':[f'inputs/{ident}/research/camera_interface.json',f'inputs/{ident}/motion_pair/static_estimated/motion.safetensors'],
        'conditioning':[f'frontend/{ident}/research/root_local.json'],
        'generate':[f'frontend/{ident}/outputs/turbo/metadata.json'],
        'validate_generation':[f'frontend/{ident}/research/generated_validation.json'],
        'generated_preview':[f'previews/{ident}/generated_multiview.mp4'],
        'comparison':[f'previews/{ident}/media_validation.json'],
        'masks':[f'backend/research/{ident}-mask-ready-complete.json'],
        'hull':[f'rgba/{ident}/init.npz'],
        'manifest':[f'backend/four_dg/data/{ident}/fdanyone.json'],
        'geometry':[f'backend/four_dg/data/{ident}/geometry_audit.json'],
        'configure':[f'backend/configs/{ident}_33k.json'],
        'native':[f'backend/runs/{ident}_33k/complete.json'],
        'alpha':[f'backend/runs/{ident}_alpha/complete.json'],
        'A':[f'backend/runs/{ident}_A/complete.json'],
        'package':[f'assets/{ident}/package_manifest.json'],
        'replay':[f'previews/{ident}/independent_replay.json'],
    }[name]
    assert all((ROOT/p).is_file() for p in outputs),('Missing expected products',outputs)
    record.write_text(json.dumps(dict(stage=name,finished=time.time(),selection_sha256=sha(ROOT/'SELECTED_TRACKS.json'),
        baseline_sha256=sha(ROOT/'BASELINE_LOCK.json'),outputs={p:sha(ROOT/p) for p in outputs}),indent=2)+'\n')

async def person(ident,phase):
    choice=selected(ident);row=STATE['people'][ident]
    if choice['camera_class'] not in ('static_verified','approximately_static'):
        row.update(state='camera_interface_not_applicable_pending',reason='Original source moving or unknown; no static assumption applied');save();return
    try:
        if phase=='prepare':
            await stage(ident,'prepare',False)
            await stage(ident,'segment',True,12000)
            await stage(ident,'motion',True,10000)
            row.update(state='pose_source_review_required');save();return
        if phase=='asset':
            gate=json.loads((ROOT/'inputs'/ident/'research/motion_gate.json').read_text())
            assert gate['variants']['static_estimated']['allow_exploratory_generation'] and gate['visual_review']!='pending'
            await stage(ident,'conditioning',True,14000)
            await stage(ident,'generate',True,35000)
            await stage(ident,'validate_generation',False)
            await stage(ident,'generated_preview',False)
            await stage(ident,'masks',True,14000)
            await stage(ident,'hull',False)
            await stage(ident,'manifest',False)
            await stage(ident,'geometry',False)
            await stage(ident,'configure',False)
            await stage(ident,'native',True,24000)
            await stage(ident,'alpha',True,24000)
            await stage(ident,'A',True,24000)
            await stage(ident,'package',False)
            await stage(ident,'replay',True,10000)
            await stage(ident,'comparison',False)
            row.update(state='asset_replayed_quality_unreviewed');save()
    except Exception as error:
        row.update(state='failed',error=repr(error),failed_at=time.time());save()

async def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['prepare','asset'],required=True);parser.add_argument('ids',nargs='+');args=parser.parse_args()
    assert len(set(args.ids))==len(args.ids)
    # This controller is single-instance; independent people are async children.
    import fcntl
    lock=(ROOT/'queue.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    STATE.update(state='running',phase=args.phase,pid=os.getpid(),started_at=time.time(),people={i:{'state':'pending'} for i in args.ids});save()
    await asyncio.gather(*(person(i,args.phase) for i in args.ids))
    STATE.update(state='phase_finished',finished_at=time.time());save()

if __name__=='__main__':asyncio.run(main())
