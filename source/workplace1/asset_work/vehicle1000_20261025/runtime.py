"""Shared GPU leases and resumable publication for this vehicle production run."""
import asyncio, fcntl, importlib.util, os, shutil, sys, time
from pathlib import Path
from collections import Counter
P=Path('/mnt/ojc/pre_flux_pipeline'); ROOT=Path(__file__).resolve().parent
LIBRARY=Path('/mnt/ojc/workplace1/asset_library')
sys.path.insert(0,str(P))
from asset_extraction import generation
from asset_extraction.generation import sha256, prepare_jobs, generate
from asset_extraction.export import curate,export_library
from asset_extraction.io_utils import read_json,write_json
GPUS=[0,1,2,3,4,5,7]  # GPU 6 hosts the shared Qwen service.

class SharedGpuPool:
    def __init__(self,gpus,**kwargs):
        self.gpus=list(gpus);self.held={};self.lock=asyncio.Lock()
        (ROOT/'locks').mkdir(parents=True,exist_ok=True)
    async def acquire(self,stage):
        required={'flux':24,'H':12,'B0':8,'render':8,'frontend':8}[stage]*1024
        last=0
        while True:
            async with self.lock:
                free=await generation.gpu_free_mib()
                for gpu in sorted(self.gpus,key=lambda g:-free.get(g,0)):
                    if gpu in self.held or free.get(gpu,0)<required:continue
                    handle=(ROOT/'locks'/f'gpu{gpu}.lock').open('a')
                    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:handle.close();continue
                    self.held[gpu]=handle
                    print(f'GPU lease: stage={stage} gpu={gpu} free_MiB={free[gpu]}',flush=True)
                    return gpu
            if time.monotonic()-last>60:
                print(f'Waiting for shared GPU: stage={stage} free_MiB={free}',flush=True);last=time.monotonic()
            await asyncio.sleep(5)
    def release(self,gpu):
        handle=self.held.pop(gpu);fcntl.flock(handle,fcntl.LOCK_UN);handle.close()

generation.GpuPool=SharedGpuPool

def publisher(batch_id):
    spec=importlib.util.spec_from_file_location('vehicle_publisher', '/mnt/ojc/workplace1/asset_work/batch20_20261024/scripts/run_batch.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.BATCH_ID=batch_id
    return module

def native_identity(row,dataset):
    from identity import canonical_identity
    provenance=row['provenance'];direct=provenance.get('production_identity') or provenance['source'].get('production_identity')
    if direct:return canonical_identity(direct,dataset,provenance['scene_id'])
    path=ROOT/'identity'/f'{dataset}_{provenance["scene_id"]}.json'
    if path.exists():
        oid=provenance.get('source_obj_id',row['uid'])
        return canonical_identity(read_json(path)['matches'].get(oid,{}),dataset,provenance['scene_id'])
    return {}

def apply_identity_curation(state,dataset,batch_id):
    """Original visual verdict stays recorded; proven duplicate assets are rejected separately."""
    current={f'{batch_id}:{dataset}:{uid}' for uid in state['assets']}
    known={}
    for row in read_json(LIBRARY/'manifests/asset_library_rigid.json')['assets']:
        if row['dataset']!=dataset or row['uid'] in current or row['status'] not in {'keep','review'}:continue
        # Object IDs are only comparable inside their own production run.
        info=native_identity(row,dataset) if row['provenance'].get('batch_id')==batch_id else row['provenance'].get('production_identity',{})
        if info.get('native_id'):known.setdefault(info['native_id'],row['uid'])
    groups={}
    for uid,result in state['assets'].items():
        if 'entry' not in result:continue
        entry=result['entry'];info=native_identity(entry,dataset)
        entry['provenance']['production_identity']=info
        result.setdefault('visual_curation_status',result['status'])
        if result['status'] not in {'keep','review'}:continue
        native=info.get('native_id')
        if info.get('status')=='conflicting_native_identities':
            result['identity_rejection']=dict(reason='conflicting_native_identities',evidence=info.get('evidence'))
        elif native in known:
            result['identity_rejection']=dict(reason='duplicate_native_identity',native_id=native,duplicate_of=known[native])
        elif native:groups.setdefault(native,[]).append((uid,result))
    for native,group in groups.items():
        winner=max(group,key=lambda pair:(pair[1]['status']=='keep',pair[1]['entry']['quality_score'],len(pair[1]['entry']['provenance']['real_images'])))
        for uid,result in group:
            if uid!=winner[0]:result['identity_rejection']=dict(reason='duplicate_native_identity',native_id=native,duplicate_of=f'{batch_id}:{dataset}:{winner[0]}')
    for result in state['assets'].values():
        if result.get('identity_rejection'):
            result['status']='reject';result['entry']['status']='reject';result['entry']['identity_rejection']=result['identity_rejection']
    state['counts']=dict(queries=len(state['assets']),**dict(Counter(r['status'] for r in state['assets'].values())))
    state['identity_policy']='dataset-native identity filter; original visual quality decision retained separately'
    return state

def discard_rejected(chunk,state):
    """Only this run's rejected outputs; archive the complete verdict before unlinking."""
    from asset_extraction.export import _identity_variation_only, VEHICLES
    archive=chunk/'rejection_archive.json'
    records={};protected={}
    for uid,result in state['assets'].items():
        if result['status']!='reject':continue
        entry=result.get('entry',{});quality=entry.get('quality',{})
        if (entry.get('category') in VEHICLES and not result.get('identity_rejection')
                and quality.get('audit') and quality.get('rendering')
                and _identity_variation_only(quality['audit'],quality['rendering'])):
            protected[uid]=dict(reason='identity_difference_only',
                canonical_output_present=Path(entry['canonical']['canonical_path']).is_file())
        else:records[uid]=result
    if protected:write_json(chunk/'protected_identity_variations.json',
        dict(policy='vehicle_identity_variation_retained_v1',assets=protected,
             note='Existing outputs are protected. Previously deleted outputs are not recreated by cleanup.'))
    if not records:return
    generated=read_json(chunk/'generation/generation.json')
    targets=[]
    for row in generated['assets']:
        if row['obj_id'] in records:
            targets += [chunk/'generation'/f'asset_{row["item_index"]:05d}',chunk/'curation'/f'asset_{row["item_index"]:05d}']
    if (chunk/'library/manifests/asset_library_rigid.json').exists():
        all_rows=read_json(chunk/'library/manifests/asset_library_rigid.json')['assets']
        protected={r['canonical']['canonical_path'] for r in all_rows if r['uid'] not in records}
        targets += [chunk/'library'/r['canonical']['canonical_path'] for r in all_rows
                    if r['uid'] in records and r['canonical']['canonical_path'] not in protected]
    if (chunk/'retained_library/manifests/asset_library_rigid.json').exists():
        rows=read_json(chunk/'retained_library/manifests/asset_library_rigid.json')['assets']
        protected={r['canonical']['canonical_path'] for r in rows if r['uid'] not in records}
        targets += [chunk/'retained_library'/r['canonical']['canonical_path'] for r in rows
                    if r['uid'] in records and r['canonical']['canonical_path'] not in protected]
    files=[p for t in targets for p in (t.rglob('*') if t.is_dir() else [t]) if p.is_file()]
    previous=read_json(archive) if archive.exists() else dict(records={},files=[])
    old_files={r['path']:r for r in previous['files']}
    old_files.update({str(p):dict(path=str(p),bytes=p.stat().st_size,sha256=sha256(p)) for p in files})
    receipt=dict(records={**previous['records'],**records},files=list(old_files.values()))
    write_json(archive,dict(receipt,status='archived_before_cleanup'))
    for target in targets:
        if not target.resolve().is_relative_to(chunk.resolve()):raise ValueError('Cleanup escaped its own chunk')
        if target.is_dir():shutil.rmtree(target)
        elif target.is_file():target.unlink()
    write_json(archive,dict(receipt,status='outputs_deleted',completed=time.time()))

async def process_chunk(chunk,rows,routing,config,dataset,batch_id,gpus=GPUS):
    chunk=Path(chunk);chunk.mkdir(parents=True,exist_ok=True)
    done=chunk/'completed.json'
    if done.exists():
        result=read_json(done)
        for p in result['published']:
            if sha256(p['canonical_path'])!=p['sha256']:raise ValueError('Published asset changed')
        discard_rejected(chunk,read_json(chunk/'curation/curation.json'))
        return result
    if shutil.disk_usage(chunk).free<20*1024**3:raise OSError('Under 20 GiB free; preserve progress and release space')
    hf=write_json(chunk/'flux_input.json',rows)
    rf=write_json(chunk/'asset_routes.json',dict(routing,flux_input=str(hf),flux_input_sha256=sha256(hf)))
    bundle=prepare_jobs(hf,rf,read_json(P/'configs/generation.example.json'))
    stage=chunk/'curation/curation.json'
    state=read_json(stage) if stage.exists() else None
    if not state or 'counts' not in state:
        await generate(bundle,chunk/'generation',gpus)
        state=await curate(chunk/'generation/generation.json',chunk/'library',config,
            renderer_python='/home/wmj/miniconda3/envs/sim/bin/python',gpus=gpus,work_dir=chunk/'curation')
    if batch_id==ROOT.name:
        state=apply_identity_curation(state,dataset,batch_id)
        write_json(stage,state)
    retained=[r['entry'] for r in state['assets'].values() if r['status'] in {'keep','review'}]
    if retained and not (chunk/'retained_library').exists():export_library(retained,chunk/'retained_library')
    pub=publisher(batch_id).publish(chunk/'retained_library',dataset,chunk.name) if retained else []
    generated=read_json(chunk/'generation/generation.json')
    result=dict(dataset=dataset,queries=len(rows),counts=state['counts'],generation_counts=generated['counts'],published=pub,
        kept=sum(r['status']=='keep' for r in pub),completed=time.time())
    # Commit completion before cleanup so resumption never regenerates deleted rejected payloads.
    write_json(done,result);discard_rejected(chunk,state)
    print('COMPLETED',str(chunk),state['counts'],flush=True)
    return result
