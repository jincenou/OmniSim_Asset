"""Shared GPU leases and resumable publication for this vehicle production run."""
import asyncio, fcntl, importlib.util, os, shutil, sys, time
from pathlib import Path
from collections import Counter
P=Path('/mnt/ojc/DriveLab'); ROOT=Path(__file__).resolve().parent
LIBRARY=Path('/mnt/ojc/workplace1/asset_library')
sys.path.insert(0,str(P))
from asset_extraction import generation
from asset_extraction.generation import sha256, prepare_jobs, generate
from asset_extraction.export import curate,export_library
from asset_extraction.io_utils import read_json,write_json
GPUS=list(range(8))  # Qwen replicas reserve their cards through the same GPU leases.

class SharedGpuPool:
    def __init__(self,gpus,**kwargs):
        self.gpus=list(gpus);self.held={};self.lock=asyncio.Lock()
        self.lease_dir=LIBRARY/'.gpu_locks';self.lease_dir.mkdir(parents=True,exist_ok=True)
    async def acquire(self,stage):
        required={'flux':24,'H':12,'B0':8,'render':8,'frontend':8}[stage]*1024
        last=0
        while True:
            async with self.lock:
                try:
                    free=await generation.gpu_free_mib()
                except asyncio.TimeoutError:
                    # No memory reading means no admission. A transient probe
                    # timeout should not turn an otherwise valid asset into a failure.
                    print(f'GPU memory probe timed out; retrying stage={stage}',flush=True)
                    await asyncio.sleep(5)
                    continue
                for gpu in sorted(self.gpus,key=lambda g:-free.get(g,0)):
                    if gpu in self.held or free.get(gpu,0)<required:continue
                    handle=(self.lease_dir/f'gpu{gpu}.lock').open('a')
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

def native_identity(row,dataset):
    from identity import canonical_identity
    provenance=row['provenance'];direct=provenance.get('production_identity') or provenance['source'].get('production_identity')
    if direct:return canonical_identity(direct,dataset,provenance['scene_id'])
    path=ROOT/'identity'/f'{dataset}_{provenance["scene_id"]}.json'
    if path.exists():
        oid=provenance.get('source_obj_id',row['uid'])
        return canonical_identity(read_json(path)['matches'].get(oid,{}),dataset,provenance['scene_id'])
    return {}

def apply_identity_curation(state,dataset,batch_id,protected_ids=()):
    """Original visual verdict stays recorded; proven duplicate assets are rejected separately."""
    current={f'{batch_id}:{dataset}:{uid}' for uid in state['assets']}
    current.update(f"{dataset}:{result['entry']['provenance']['scene_id']}:{uid}"
        for uid,result in state['assets'].items() if 'entry' in result)
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
        winner=max(group,key=lambda pair:(pair[0] in protected_ids,pair[1]['status']=='keep',pair[1]['entry']['quality_score'],len(pair[1]['entry']['provenance']['real_images'])))
        for uid,result in group:
            if uid!=winner[0]:result['identity_rejection']=dict(reason='duplicate_native_identity',native_id=native,duplicate_of=f'{batch_id}:{dataset}:{winner[0]}')
    for result in state['assets'].values():
        if result.get('identity_rejection'):
            result['status']='reject';result['entry']['status']='reject';result['entry']['identity_rejection']=result['identity_rejection']
    state['counts']=dict(queries=len(state['assets']),**dict(Counter(r['status'] for r in state['assets'].values())))
    state['identity_policy']='dataset-native identity filter; original visual quality decision retained separately'
    return state

def preserve_outputs(chunk,state):
    write_json(chunk/'retention.json',dict(policy='retain_all_inputs_intermediates_and_outputs',
        retained_statuses=['keep','review','reject','failed'],automatic_deletion=False,
        reason='User requests reuse for later cross-model comparisons',updated=time.time()))

def minimum_free_gib():
    """Read the user-controlled admission threshold for every new chunk."""
    import math
    value=float(read_json(ROOT/'retention_policy.json')['minimum_free_gib_for_new_chunk'])
    if not math.isfinite(value) or value<=0:
        raise ValueError('minimum_free_gib_for_new_chunk must be positive and finite')
    return value

def reconstruction_gpus(gpus):
    """Shard by reconstruction slots, excluding cards reserved for Qwen."""
    path=ROOT/'vlm_process.json'
    reserved=set(read_json(path).get('replica_gpus',[])) if path.exists() else set()
    workers=[gpu for gpu in gpus if gpu not in reserved]
    if not workers:
        raise ValueError('No generation GPU slots remain after Qwen reservations')
    return workers

async def process_chunk(chunk,rows,routing,config,dataset,batch_id,gpus=GPUS):
    from asset_extraction.bank import publish
    chunk=Path(chunk);chunk.mkdir(parents=True,exist_ok=True)
    done=chunk/'completed.json'
    if done.exists():
        result=read_json(done)
        for p in result['published']:
            if sha256(p['canonical_path'])!=p['sha256']:raise ValueError('Published asset changed')
        preserve_outputs(chunk,read_json(chunk/'curation/curation.json'))
        return result
    minimum=minimum_free_gib()
    if shutil.disk_usage(chunk).free<minimum*1024**3:
        raise OSError(f'Under {minimum:g} GiB free; preserving state until storage is available')
    worker_gpus=reconstruction_gpus(gpus)
    hf=write_json(chunk/'flux_input.json',rows)
    rf=write_json(chunk/'asset_routes.json',dict(routing,flux_input=str(hf),flux_input_sha256=sha256(hf)))
    bundle=prepare_jobs(hf,rf,read_json(P/'configs/generation.example.json'))
    stage=chunk/'curation/curation.json'
    state=read_json(stage) if stage.exists() else None
    if not state or 'counts' not in state:
        await generate(bundle,chunk/'generation',worker_gpus)
        state=await curate(chunk/'generation/generation.json',chunk/'library',config,
            renderer_python='/home/wmj/miniconda3/envs/sim/bin/python',gpus=gpus,work_dir=chunk/'curation')
    from asset_extraction.quality import repair_shapes
    initial=chunk/'curation/initial_curation.json'
    if not initial.exists():write_json(initial,state)
    state,generated=await repair_shapes(chunk/'generation/generation.json',read_json(initial),config,
        chunk/'shape_retry',renderer_python='/home/wmj/miniconda3/envs/sim/bin/python',gpus=worker_gpus)
    write_json(chunk/'selected_generation.json',generated)
    state=apply_identity_curation(state,dataset,batch_id)
    for result in state['assets'].values():
        if 'entry' in result:result['entry']['provenance']['batch_id']=batch_id
    write_json(stage,state)
    published=publish(state,LIBRARY,dataset=dataset,scene_id=rows[0]['scene_id'])
    pub=[dict(uid=r['uid'],canonical_path=str(LIBRARY/r['canonical']['canonical_path']),
        sha256=r['canonical']['sha256'],source_obj_id=r['provenance']['source_obj_id'],
        status=r['status'],category=r['category']) for r in published]
    technical=[]
    for row in generated['assets']:
        record=state['assets'][row['obj_id']]
        render=record.get('entry',{}).get('quality',{}).get('rendering',{})
        if not render:
            for directory in ['canonical_render','axis_render']:
                evidence=chunk/'curation'/f'asset_{row["item_index"]:05d}'/directory/'render.json'
                if evidence.exists():render=read_json(evidence);break
        pixels=render.get('foreground_pixels',[])
        technical.append(dict(obj_id=row['obj_id'],category=row['category'],generated=row['status']=='generated',
            valid_16=bool(row['status']=='generated' and row.get('gaussian_count',0)>0 and render.get('all_16_nonempty')
                and len(pixels)==16 and all(v>0 for v in pixels)),status=record['status'],rendering=render))
    write_json(chunk/'technical_validity.json',dict(queries=len(rows),assets=technical))
    result=dict(dataset=dataset,queries=len(rows),counts=state['counts'],generation_counts=generated['counts'],published=pub,
        selected_generation=str(chunk/'selected_generation.json'),quality_policy='insertion_quality_shape_retry_v2',
        kept=sum(r['status']=='keep' for r in pub),completed=time.time())
    write_json(done,result)
    preserve_outputs(chunk,state)
    print('COMPLETED',str(chunk),state['counts'],flush=True)
    return result


async def upgrade_completed_chunks(datasets=('nuscenes','waymo')):
    """Re-audit old rejects before resuming; preserve published packages and all old receipts."""
    from copy import deepcopy
    from asset_extraction.quality import POLICY,reassess,repair_shapes
    from asset_extraction.bank import publish
    from asset_extraction.config import load_config
    from asset_extraction.real_views import VEHICLES

    async def dataset_upgrade(dataset):
        path=ROOT/f'{dataset}_status.json'
        if not path.exists():return
        progress=read_json(path)
        progress.update(status='quality_policy_upgrade',updated=time.time());write_json(path,progress)
        for name in progress['chunks']:
            chunk=Path(name);receipt=read_json(chunk/'completed.json')
            if receipt.get('quality_policy')==POLICY:continue
            work=chunk/'quality_v2';work.mkdir(exist_ok=True)
            for source,target in [('completed.json','previous_completed.json'),
                                  ('curation/curation.json','previous_curation.json'),
                                  ('technical_validity.json','previous_technical_validity.json')]:
                if not (work/target).exists() and (chunk/source).exists():
                    shutil.copyfile(chunk/source,work/target)
            old=read_json(work/'previous_curation.json')
            previous=read_json(work/'previous_completed.json')
            scene=read_json(chunk/'flux_input.json')[0]['scene_id']
            config=load_config(ROOT/'configs'/f'{dataset}_{scene}.yaml')
            generation_path=chunk/'generation/generation.json'
            reviewed=await reassess(old,read_json(generation_path),config,work/'reassessed.json')
            final,selected=await repair_shapes(generation_path,reviewed,config,work/'shape_retry',
                renderer_python='/home/wmj/miniconda3/envs/sim/bin/python',gpus=GPUS)
            existing={r['source_obj_id'] for r in previous['published']}
            final=apply_identity_curation(final,dataset,ROOT.name,protected_ids=existing)
            # Published assets remain immutable, including their metadata and receipt signatures.
            for uid in existing:final['assets'][uid]=deepcopy(old['assets'][uid])
            for result in final['assets'].values():
                if 'entry' in result:result['entry']['provenance']['batch_id']=ROOT.name
            final['counts']=dict(queries=len(final['assets']),**Counter(r['status'] for r in final['assets'].values()))
            additions=publish(dict(assets={uid:r for uid,r in final['assets'].items() if uid not in existing}),
                LIBRARY,dataset=dataset,scene_id=scene)
            published=previous['published']+[dict(uid=r['uid'],canonical_path=str(LIBRARY/r['canonical']['canonical_path']),
                sha256=r['canonical']['sha256'],source_obj_id=r['provenance']['source_obj_id'],status=r['status'],
                category=r['category']) for r in additions]
            write_json(chunk/'selected_generation.json',selected)
            write_json(chunk/'curation/curation.json',final)
            technical=[]
            for row in selected['assets']:
                record=final['assets'][row['obj_id']];render=record.get('entry',{}).get('quality',{}).get('rendering',{})
                if not render:
                    for directory in ['canonical_render','axis_render']:
                        candidate=chunk/'curation'/f'asset_{row["item_index"]:05d}'/directory/'render.json'
                        if candidate.exists():render=read_json(candidate);break
                pixels=render.get('foreground_pixels',[])
                technical.append(dict(obj_id=row['obj_id'],category=row['category'],generated=row['status']=='generated',
                    valid_16=bool(row['status']=='generated' and row.get('gaussian_count',0)>0 and
                        render.get('all_16_nonempty') and len(pixels)==16 and all(v>0 for v in pixels)),
                    status=record['status'],rendering=render,selected_attempt=row.get('selected_attempt',0)))
            write_json(chunk/'technical_validity.json',dict(queries=len(technical),assets=technical))
            receipt.update(counts=final['counts'],published=published,kept=sum(r['status']=='keep' for r in published),
                generation_counts=selected['counts'],selected_generation=str(chunk/'selected_generation.json'),
                quality_policy=POLICY,previous_receipt=str(work/'previous_completed.json'),policy_updated=time.time())
            write_json(chunk/'completed.json',receipt)
            completed=[read_json(Path(c)/'completed.json') for c in progress['chunks']]
            progress.update(kept=sum(c['kept'] for c in completed),
                kept_other=sum(r['status']=='keep' and r['category'] not in VEHICLES for c in completed for r in c['published']),
                queries=sum(c['queries'] for c in completed),updated=time.time(),quality_policy=POLICY)
            write_json(path,progress)
            print('QUALITY_UPGRADED',chunk.name,final['counts'],flush=True)
        progress.update(status='quality_policy_ready',updated=time.time());write_json(path,progress)

    results=await asyncio.gather(*(dataset_upgrade(dataset) for dataset in datasets),return_exceptions=True)
    for result in results:
        if isinstance(result,BaseException):raise result
