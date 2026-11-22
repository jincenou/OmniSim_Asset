"""Frozen all-valid and final-keep cohorts, with full query failure accounting."""
import asyncio,copy,json,os,sys,time,importlib.util
from pathlib import Path
from collections import defaultdict
from runtime import ROOT,LIBRARY,read_json,write_json,sha256,SharedGpuPool
from asset_extraction.vision import _bbox_iou
OLD=Path('/mnt/ojc/workplace1/asset_work/batch20_20261024')
SCRIPT=ROOT/'evaluate.py'
R=Path('/mnt/ojc/workplace1/reconviagen_demo')
GROUP=os.environ.get('ASSET_METRIC_GROUP','vehicles')
if GROUP not in {'vehicles','other'}:raise ValueError('Unknown metric group')
SNAPSHOT=os.environ.get('VEHICLE_METRIC_SNAPSHOT','final_'+GROUP)
if not SNAPSHOT or any(not (c.isalnum() or c in '_-') for c in SNAPSHOT):
    raise ValueError('Snapshot name must contain only letters, digits, underscores or hyphens')
REGISTRY=ROOT/'evaluation'/f'metric_registry_{SNAPSHOT}.json'

def ground_truth_path(scene):
    original=OLD/'evaluation/ground_truth'/f'{scene}.json'
    return original if original.exists() else ROOT/'evaluation/native_registry/evaluation/ground_truth'/f'{scene}.json'

def prepare_ground_truth():
    """Extend the same native annotation protocol to every processed scene, regardless of its keep count."""
    build=ROOT/'evaluation/native_registry'
    scenes={d:[] for d in ('nuscenes','waymo')}
    for path in sorted((ROOT/'runs').glob('*/production_selection.json')):
        dataset,scene=path.parent.name.split('_',1)
        if not ground_truth_path(scene).exists():scenes[dataset].append(scene)
    registry=read_json(ROOT/'registry.json')
    selected=dict(nuscenes=scenes['nuscenes'],waymo=[r for r in registry['waymo'] if r['scene_id'] in scenes['waymo']])
    write_json(build/'registry.json',selected)
    def link(source,target):
        target.parent.mkdir(parents=True,exist_ok=True)
        if not source.is_file():raise FileNotFoundError(source)
        if not target.exists():target.symlink_to(source)
        elif target.resolve()!=source.resolve():raise ValueError(f'Annotation input changed: {target}')
    for scene in scenes['nuscenes']:
        link(ROOT/'inputs/nuscenes/debug/dataset_scenes'/f'{scene}.json',build/'runs/nuscenes10/debug/dataset_scenes'/f'{scene}.json')
    for scene in scenes['waymo']:
        link(ROOT/'inputs/waymo_indexes'/f'{scene}.json',build/'inputs/waymo_indexes'/f'{scene}.json')
        link(ROOT/'evaluation/waymo_native'/f'{scene}.jsonl',build/'evaluation/waymo_native'/f'{scene}.jsonl')
    if any(scenes.values()):
        spec=importlib.util.spec_from_file_location('production_native_registry',OLD/'scripts/prepare_evaluation.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.B=build;module.make_registry()
    write_json(build/'last_build.json',dict(created=time.time(),scenes=scenes,
        protocol_script_sha256=sha256(OLD/'scripts/prepare_evaluation.py'),
        selection='All processed scene inputs, independent of asset admission or quality outcomes. Existing first20 registries unchanged.'))
    return scenes

def freeze():
    from asset_extraction.real_views import VEHICLES
    from identity import canonical_identity
    if REGISTRY.exists():return read_json(REGISTRY)
    rows=[];entries=[];sources={};truth={};queries=0
    for completion in sorted((ROOT/'chunks').glob('*/completed.json')):
        chunk=completion.parent;dataset=read_json(completion)['dataset']
        chosen=read_json(chunk/'flux_input.json')
        wanted={r['obj_id'] for r in chosen if (r['category'] in VEHICLES)==(GROUP=='vehicles')}
        queries+=len(wanted)
        for oid,result in read_json(chunk/'curation/curation.json')['assets'].items():
            if oid not in wanted or 'entry' not in result:continue
            entry=copy.deepcopy(result['entry']);entry['dataset']=dataset;entry['status']=result['status'];entries.append(entry)
    for entry in entries:
        d=entry['dataset'];scene=entry['provenance']['scene_id'];key=(d,scene)
        if key not in sources:
            run=ROOT/'runs'/f'{d}_{scene}'
            sources[key]={r['asset_id']:r for r in read_json(run/'scenes'/scene/'logs/export_items_full.json')}
            gt=ground_truth_path(scene);truth[key]=read_json(gt)['records'] if gt.exists() else []
        oid=entry['provenance']['source_obj_id'];source=sources[key][oid];obs=source['source']
        anchors=[r for r in source.get('real_images',[]) if r['source_image']==obs['source_image'] and max(abs(a-b) for a,b in zip(r['bbox_xyxy'],obs['bbox_xyxy']))<.01]
        if len(anchors)!=1:raise ValueError('Cannot identify immutable original SAM foreground: '+oid)
        anchor=anchors[0]
        linked=sorted([(_bbox_iou(obs['bbox_xyxy'],t['bbox_xyxy']),i) for i,t in enumerate(truth[key]) if t['source_image']==obs['source_image']],reverse=True)
        native=truth[key][linked[0][1]] if linked and linked[0][0]>=.5 else None
        identity=entry['provenance'].get('production_identity',{}).get('native_id')
        if native and identity:
            linked_id=canonical_identity(dict(native_id=native['id']),d,scene)['native_id']
            if linked_id!=canonical_identity(dict(native_id=identity),d,scene)['native_id']:native=None
        eligible=bool(native and identity and native.get('pose_link') and entry['canonical']['heading_verified'])
        rows.append(dict(obj_id=oid,dataset=d,scene_id=scene,route=entry['route'],category=entry['category'],quality_status=entry['status'],
            canonical=entry['canonical'],ply=entry['canonical']['canonical_path'],ply_sha256=entry['canonical']['sha256'],
            source_rgba=anchor['rgba_path'],source_sha256=sha256(anchor['rgba_path']),source_crop_box=anchor['crop_box'],
            native_dimensions_annotation=native if native and identity and entry['canonical']['heading_verified'] else None,
            matched_annotation=native if eligible else None,matched_exclusion=None if eligible else 'Native object pose and verified heading unavailable for this anchor'))
    protocol=copy.deepcopy(read_json(OLD/'frontend_revision_01/evaluation/metric_registry.json')['protocol'])
    for item in protocol['checkpoints'].values():
        if sha256(item['path'])!=item['sha256']:raise ValueError('Metric checkpoint changed')
    protocol.update(version='scene_production_retention_accounted_v2',script_sha256=sha256(SCRIPT),snapshot_script_sha256=sha256(__file__),
        queries=queries,exported_payloads=len(rows),group=GROUP,partial_snapshot=False,created=time.time(),
        cohorts='All exported assets with curation records, including reject/review; final keep subset reported separately. Failed/no-entry queries remain in the production AOR denominator.',
        failure_policy='Original inputs and query records are retained; user-authorized cleanup removed some historical rejected PLY outputs. Exported payload count denotes recorded exports, not current file availability. Missing outputs remain in the registry and production AOR denominator. Similarity requires an actual valid render or its previously verified cache. No fabricated pixel alignment or best-view search.',
        identity_protocol='Dataset-native vehicle identity assists deduplication; other categories use RGB frontend association.',
        comparisons='Separate datasets and vehicle/nonvehicle groups are different populations, not a controlled method ranking.',
        dimensions='Verified native identity and canonical heading required. Gaussian-center 0.5%-99.5% axis extents are uniformly scaled to native cuboid length. Width/height APE are averaged per identity. Box_extent_IoU compares aligned extents at common XY center and ground, not surface geometry, physical placement, independent scale accuracy, or detection IoU. Missing native evidence is N/A.',
        FAD='Not used. User confirmed adding separately named FD-DINO on 2026-10-31.',
        FD_DINO='Frechet distance of L2-normalized final DINOv3 CLS features, using the same checkpoint and 512px foreground preprocessing as DINO-I16. One source image per identity versus 16 orbit renders per identity; no matched-source extra view. Unbiased covariance, float64 statistics, at least two source identities. Lower is better. This normalized-feature distance has a different numeric scale from FID and raw-feature FD; it is not FAD.')
    result=dict(protocol=protocol,assets=rows);write_json(REGISTRY,result);return result

async def wait_for_recovery_snapshot_lock():
    # Recovery rewrites selected attempts and curation; final cohorts must wait.
    import fcntl
    work=ROOT/'validation/ffmpeg_recovery_20261103'
    if not (work/'plan.json').exists():return None
    handle=(work/'recovery.lock').open('a')
    try:
        while True:
            try:
                fcntl.flock(handle,fcntl.LOCK_SH|fcntl.LOCK_NB)
                break
            except BlockingIOError:
                write_json(ROOT/'evaluation/final_snapshot_wait.json',dict(status='waiting_for_encoder_recovery',updated=time.time()))
                await asyncio.sleep(5)
        status=read_json(work/'status.json')
        if status.get('status') not in {'completed','completed_with_errors'}:
            raise RuntimeError('Recovery has no live lock and no terminal result; resolve before final metric freeze')
        return handle
    except BaseException:
        handle.close();raise

async def main():
    handle=await wait_for_recovery_snapshot_lock()
    try:doc=freeze()
    finally:
        if handle is not None:handle.close()
    out=ROOT/'evaluation';out.mkdir(exist_ok=True);(ROOT/'pilot_results').mkdir(exist_ok=True)
    devices=[int(value) for value in os.environ.get('VEHICLE_METRIC_GPUS','0,1,2,3,4,5,7').split(',')]
    if not devices or len(set(devices))!=len(devices):raise ValueError('Metric GPUs must be distinct')
    pool=SharedGpuPool(devices);evaluation_slot=asyncio.Semaphore(min(4,len(devices)))
    async def run_stage(python,args,name,stage='render',gpu=True):
        device=await pool.acquire(stage) if gpu else None
        env=dict(os.environ,ASSET_EVALUATION_ROOT=str(ROOT),ASSET_METRIC_REGISTRY=str(REGISTRY),ASSET_METRIC_RESULT_NAME=SNAPSHOT,
            OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            PATH=str(Path(python).parent)+':'+os.environ['PATH'],CPATH='/home/wmj/miniconda3/envs/splatad/include',MAX_JOBS='2',
            TORCH_EXTENSIONS_DIR=str(R/'cache/torch_extensions'))
        if gpu:env['CUDA_VISIBLE_DEVICES']=str(device)
        proc=None;started=time.time()
        try:
            with (out/f'{name}.log').open('a') as log:
                proc=await asyncio.create_subprocess_exec(python,str(SCRIPT),*args,env=env,stdout=log,stderr=asyncio.subprocess.STDOUT)
                if await proc.wait():raise RuntimeError(f'Metric stage failed: {name}; inspect its log')
        finally:
            if proc is not None and proc.returncode is None:proc.terminate();await proc.wait()
            if device is not None:pool.release(device)
        write_json(out/f'{name}_completion.json',dict(stage=name,seconds=time.time()-started,gpu=device,completed=time.time()))
    async def launch(python,args,name,stage='render',gpu=True):
        if gpu:
            async with evaluation_slot:
                await run_stage(python,args,name,stage,gpu)
        else:await run_stage(python,args,name,stage,gpu)
    await asyncio.gather(*(launch('/home/wmj/miniconda3/envs/sim/bin/python',['render','--indices',*map(str,range(i,len(doc['assets']),len(devices)))],f'{SNAPSHOT}_render_{i}') for i in range(min(len(devices),len(doc['assets'])))))
    feature_python='/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python'
    await asyncio.gather(*(launch(feature_python,['features','--encoder',encoder],f'{SNAPSHOT}_{encoder}',stage='H' if encoder=='dino' else 'render') for encoder in ['clip','dino','fid']),
        launch(str(R/'env_refine/bin/python'),['pixels'],f'{SNAPSHOT}_pixels'))
    await launch(feature_python,['summarize'],f'{SNAPSHOT}_summary',gpu=False)
    path=ROOT/'pilot_results'/f'{SNAPSHOT}.json';result=read_json(path)
    validate_matched_summary(result)
    write_json(path,result)
    import csv
    fields=['dataset','cohort','n_exports','CLIP_I16','DINO_I16','ED_R','PSNR_dB','SSIM','LPIPS_alex','silhouette_IoU','FID_2048','FD_DINO',
            'Width_APE_percent','Height_APE_percent','Box_extent_IoU']
    accounting=['registered_exports','excluded_invalid_render','missing_ply_at_summary','valid_cached_render_without_ply']
    with (ROOT/'pilot_results'/f'{SNAPSHOT}_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields+accounting);writer.writeheader()
        for row in result['summary']:
            writer.writerow(dict(dataset=row['dataset'],cohort=row['cohort'],n_exports=row['valid_queries'],
                **{key:row[key] for key in accounting},
                **{key:row['metrics'][key].get('mean',row['metrics'][key].get('value')) for key in fields[3:]}))
    publish_metrics(result)
    # Reuse the already validated batch=1, 512px renderer timing protocol.
    kept=[row for row in doc['assets'] if row['quality_status']=='keep']
    performance=ROOT/'evaluation'/f'performance_{GROUP}'
    performance_registry=performance/'registry.json'
    write_json(performance_registry,dict(protocol=doc['protocol'],assets=kept))
    if kept:
        benchmark=Path('/mnt/ojc/DriveLab/benchmarks/vehicles.py')
        device=await pool.acquire('render')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(device),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
            CUDA_HOME='/home/wmj/miniconda3/envs/sim',PATH='/home/wmj/miniconda3/envs/sim/bin:'+os.environ['PATH'],
            CPATH='/home/wmj/miniconda3/envs/splatad/include',
            TORCH_EXTENSIONS_DIR=str(R/'cache/torch_extensions'),MAX_JOBS='2')
        process=None
        try:
            with (performance/'render.log').open('a') as log:
                process=await asyncio.create_subprocess_exec('/home/wmj/miniconda3/envs/sim/bin/python',str(benchmark),
                    'render','--registry',str(performance_registry),'--output',str(performance),
                    env=env,stdout=log,stderr=asyncio.subprocess.STDOUT)
                if await process.wait():raise RuntimeError('Performance rendering failed; see '+str(performance/'render.log'))
        finally:
            if process is not None and process.returncode is None:process.terminate();await process.wait()
            pool.release(device)
        with (performance/'summary.log').open('a') as log:
            process=await asyncio.create_subprocess_exec(sys.executable,str(benchmark),'summarize',
                '--registry',str(performance_registry),'--output',str(performance),stdout=log,stderr=asyncio.subprocess.STDOUT)
            if await process.wait():raise RuntimeError('Performance summary failed')
        import shutil
        for name in ['results.json','summary.csv','protocol.json']:
            shutil.copyfile(performance/name,LIBRARY/'previews'/ROOT.name/f'performance_{GROUP}_{name}')
    write_json(ROOT/'evaluation'/f'{SNAPSHOT}_completed.json',dict(assets=len(doc['assets']),result=str(path),finished=time.time()))

def validate_matched_summary(result):
    """Apply frozen pose exclusions without changing reusable orbit features or raw evidence."""
    path=ROOT/'evaluation'/f'{SNAPSHOT}_matched_exclusions.json'
    if not path.exists():return
    exclusions=read_json(path)
    original=ROOT/'pilot_results'/f'{SNAPSHOT}_before_alignment_validation.json'
    if not original.exists():write_json(original,result)
    keys=['ED_R','PSNR_dB','SSIM','LPIPS_alex','silhouette_IoU']
    for row in result['assets']:
        if row['obj_id'] in exclusions['objects']:
            row.update({key:None for key in keys})
            row['matched_exclusion']=exclusions['objects'][row['obj_id']]
    for summary in result['summary']:
        rows=[r for r in result['assets'] if r['dataset']==summary['dataset'] and r['valid_16_views']
              and (summary['cohort']!='final_keep' or r['quality_status']=='keep')]
        for key in keys:
            values=[r[key] for r in rows if r.get(key) is not None]
            summary['metrics'][key]=dict(n=len(values),mean=sum(values)/len(values) if values else None)
    result['protocol']['matched_pose_exclusions']=exclusions

def publish_metrics(result):
    import html,shutil
    out=LIBRARY/'previews'/ROOT.name;out.mkdir(parents=True,exist_ok=True)
    write_json(out/f'{SNAPSHOT}.json',result)
    shutil.copyfile(ROOT/'pilot_results'/f'{SNAPSHOT}_summary.csv',out/f'{SNAPSHOT}.csv')
    page='<meta charset="utf-8"><title>Asset metrics</title><h1>'+html.escape(SNAPSHOT)+'</h1><p>All valid exports and final keep are separate cohorts. Source-conditioned diagnostics; not held-out reconstruction or scene insertion quality. FD-DINO uses L2-normalized final CLS features, one source versus 16 renders per identity; lower is better. Its scale differs from FID; it is not FAD.</p><p>Historical cleanup can leave missing PLY files. Registered exports and excluded renders remain visible; missing PLY may overlap with a previously verified valid render cache. Available-render metrics can have selection bias. These counts do not replace production AOR query denominators.</p><table border="1"><tr><th>Dataset</th><th>Cohort</th><th>Valid renders</th><th>Registered exports</th><th>Excluded renders</th><th>Missing PLY</th><th>Valid cache without PLY</th>'
    keys=['CLIP_I16','DINO_I16','ED_R','PSNR_dB','SSIM','LPIPS_alex','silhouette_IoU','FID_2048','FD_DINO',
          'Width_APE_percent','Height_APE_percent','Box_extent_IoU']
    page+=''.join('<th>'+k+'</th>' for k in keys)+'</tr>'
    for row in result['summary']:
        values=[row['dataset'],row['cohort'],str(row['valid_queries'])]
        values.extend(str(row.get(key,'N/A')) for key in ['registered_exports','excluded_invalid_render','missing_ply_at_summary','valid_cached_render_without_ply'])
        for key in keys:
            m=row['metrics'][key];v=m.get('mean',m.get('value'));values.append('N/A' if v is None else f'{v:.4f}')
        page+='<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>'
    page+='</table><p><a href="'+SNAPSHOT+'.csv">CSV</a> | <a href="'+SNAPSHOT+'.json">Per-asset results and protocol</a></p>'
    (out/f'{SNAPSHOT}.html').write_text(page)
    write_json(out/'metrics_latest.json',dict(html=SNAPSHOT+'.html',assets=len(result['assets'])))

def production_ledger():
    """Construction-time AOR and annotation coverage; never equate curation with renderability."""
    from collections import Counter
    from identity import canonical_identity
    from asset_extraction.real_views import VEHICLES
    import html,csv,io
    bank=[r for r in read_json(LIBRARY/'manifests/asset_library_rigid.json')['assets']
          if r['provenance'].get('batch_id')==ROOT.name]
    attempts=[]
    for completion in sorted((ROOT/'chunks').glob('*/completed.json')):
        chunk=completion.parent;done=read_json(completion)
        rows=read_json(chunk/'flux_input.json')
        generation_path=Path(done.get('selected_generation',chunk/'generation/generation.json'))
        generation={r['obj_id']:r for r in read_json(generation_path)['assets']}
        curated=read_json(chunk/'curation/curation.json')['assets']
        if done['queries']!=len(rows):raise ValueError('Committed query count changed')
        for query in rows:
            oid=query['obj_id'];gen=generation[oid];audit=curated[oid];entry=audit.get('entry',{})
            render=entry.get('quality',{}).get('rendering',{})
            validation=entry.get('canonical',{}).get('validation',{})
            # A model-service failure may happen after a valid export/render.
            if not render:
                work=chunk/'curation'/f'asset_{gen["item_index"]:05d}'
                for path in [work/'canonical_render/render.json',work/'axis_render/render.json']:
                    if path.exists():render=read_json(path);break
            pixels=render.get('foreground_pixels',[])
            renderable=bool(gen['status']=='generated' and gen.get('gaussian_count',0)>0
                and render.get('all_16_nonempty') and len(pixels)==16 and all(p>0 for p in pixels)
                and (not validation or validation.get('gaussian_count',0)>0))
            attempts.append(dict(dataset=done['dataset'],scene_id=query['scene_id'],obj_id=oid,
                chunk=chunk.name,category=query['category'],generated=gen['status']=='generated',valid_at_construction=renderable,
                status=audit['status'],visual_status=audit.get('visual_curation_status',audit['status']),
                selected_attempt=gen.get('selected_attempt',0),quality_policy=done.get('quality_policy','original'),
                identity_rejection=audit.get('identity_rejection'),error=audit.get('error'),
                evidence_sha256=render.get('ply_sha256'),rendered_views=len(pixels),
                rejected_outputs_deleted=bool(audit['status']=='reject' and (chunk/'rejected_output_cleanup_20261102.json').exists() and not Path(entry.get('canonical',{}).get('canonical_path','/nonexistent')).is_file())))
    summaries=[]
    for dataset in ['nuscenes','waymo']:
        for family in ['vehicles','other']:
            rows=[r for r in attempts if r['dataset']==dataset and (r['category'] in VEHICLES)==(family=='vehicles')];n=len(rows)
            summaries.append(dict(dataset=dataset,group=family,completed_chunk_queries=n,generated=sum(r['generated'] for r in rows),
                render_valid=sum(r['valid_at_construction'] for r in rows),
                AOR_percent=100*sum(r['valid_at_construction'] for r in rows)/n if n else None,
                counts=dict(Counter(r['status'] for r in rows)),
                bank_acceptance_percent=100*sum(r['status']=='keep' for r in rows)/n if n else None,
                identity_rejections=sum(bool(r['identity_rejection']) for r in rows)))
    scenes=[]
    for path in sorted((ROOT/'runs').glob('*/production_selection.json')):
        run=path.parent;dataset,scene=run.name.split('_',1)
        gt_path=ground_truth_path(scene)
        if not gt_path.exists():
            scenes.append(dict(dataset=dataset,scene_id=scene,coverage_status='native_annotation_registry_pending'))
            continue
        gt=read_json(gt_path);native=[r for r in gt['records'] if r['category'] in VEHICLES]
        canonical=lambda value:canonical_identity(dict(native_id=value),dataset,scene)['native_id']
        denominator={canonical(r['id']) for r in native}
        ids=read_json(ROOT/'identity'/f'{dataset}_{scene}.json')['matches']
        handoff=[r for r in read_json(run/'handoff/flux_input.json') if r['category'] in VEHICLES];selected=read_json(path)
        admitted={canonical(r['native_id']) for r in ids.values() if r.get('native_id')}&denominator
        kept=[r for r in bank if r['dataset']==dataset and r['provenance']['scene_id']==scene and r['status']=='keep' and r['category'] in VEHICLES]
        kept_ids={canonical(r['provenance']['production_identity']['native_id']) for r in kept
                  if r['provenance'].get('production_identity',{}).get('native_id')}&denominator
        status=read_json(run/'scenes'/scene/'status.json')
        info=read_json(ROOT/f'{dataset}_status.json')['scenes'].get(scene,{})
        scenes.append(dict(dataset=dataset,scene_id=scene,coverage_status='provisional' if info.get('status')!='completed' else 'scene_completed',
            denominator_annotated_vehicle_ids=len(denominator),raw_annotation_ids=len({r['id'] for r in native}),
            admitted_linked_ids=len(admitted),kept_linked_ids=len(kept_ids),kept_total=len(kept),
            kept_without_verified_annotation_link=len(kept)-len(kept_ids),
            source_admission_coverage_percent=100*len(admitted)/len(denominator) if denominator else None,
            kept_coverage_percent=100*len(kept_ids)/len(denominator) if denominator else None,
            source_images=status['num_images'],detections=status['num_raw_detections'],tracklets=status['num_tracklets'],
            source_vlm_input=status.get('vlm',{}).get('input_count'),
            source_schema_or_api_failed=status.get('vlm',{}).get('failed_count'),
            pre_identity_handoff_queries=len(handoff),scheduled_queries=len(selected['rows']),
            pre_generation_removed=len(selected.get('removed',[])),
            reference_denominator_sha256=sha256(gt_path),identity_map_sha256=sha256(ROOT/'identity'/f'{dataset}_{scene}.json'),
            annotation_scope=gt['scope'],
            identity_scope='scene-native instance tokens' if dataset=='nuscenes' else
                'camera-scoped native tracks merged only by verified projected-LiDAR aliases; residual cross-camera duplicates can remain',
            denominator_ids=sorted(denominator),admitted_ids=sorted(admitted),kept_ids=sorted(kept_ids)))
    conditioning=[]
    for selected_path in sorted((ROOT/'runs').glob('*/production_selection.json')):
        dataset,scene=selected_path.parent.name.split('_',1)
        selection=read_json(selected_path)
        routes={r['obj_id']:r for r in selection['routes']}
        original={r['obj_id']:r for r in read_json(selected_path.parent/'handoff/asset_routes.json')['assets']}
        all_rows=selection['rows']+[r for r in read_json(selected_path.parent/'handoff/flux_input.json') if r['category'] not in VEHICLES]
        for row in all_rows:
            route=routes.get(row['obj_id'],original[row['obj_id']])
            source=route['source']
            conditioning.append(dict(dataset=dataset,scene_id=scene,obj_id=row['obj_id'],category=row['category'],
                source_foreground=row['image_path'],prompts=row['prompts'],route=route['route'],
                source=source,real_references=route['real_images'],selection=route.get('real_view_selection'),
                native_identity=source.get('production_identity'),
                flux_search_pattern=str(ROOT/'chunks'/f'{dataset}_{scene}_*'/'generation'/'asset_*'/'flux_*.png'),
                note='All input-admitted nonvehicles retained here even if the production quota later leaves them unattempted. This is an input registry, not an output-success table.'))
    write_json(ROOT/'evaluation/common_conditioning_registry.json',dict(selection='input only, before outcome filtering',assets=conditioning))
    definitions=dict(
        AOR='At construction time: queries with a decoded nonempty Gaussian and 16 nonempty orbit views / all queries in committed chunks, including generation or render failures. Visual quality rejection and duplicate rejection do not erase technical validity.',
        AOR_scope='Operational query-attempt rate. Vehicle queries are native-identity deduplicated where the linkage is reliable; unmatched identities retain an explicit uncertainty. Pending chunks are excluded from this snapshot, not counted as failures.',
        coverage='Unique verified native vehicle IDs with a kept asset / fixed native annotated vehicle IDs in the processed scene frames. Unmatched kept assets do not increase this numerator.',
        coverage_limits='nuScenes uses in-frustum cuboids, including occluded objects; Waymo uses camera annotations and only verified cross-camera links. Different annotation scopes make the two percentages non-comparable as algorithm rankings. Not coverage of all unannotated scene objects.',
        source_failures='Schema/API errors are distinct from source-quality rejection. Historical failures remain recorded; later frontend fixes do not rewrite them.',
        missing_metrics='FID, separately named FD-DINO (L2-normalized final CLS), and other similarity metrics are reported separately for all valid exports and final keep assets, by dataset and vehicle/nonvehicle group. Failed renders remain in AOR but have no similarity score. FD-DINO was explicitly approved by the user on 2026-10-31; it is not called FAD.')
    result=dict(updated_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),summaries=summaries,
        scenes=scenes,attempts=attempts,definitions=definitions,scene_registry=str(ROOT/'registry.json'))
    out=LIBRARY/'previews'/ROOT.name
    write_json(ROOT/'evaluation/production_ledger.json',result);write_json(out/'production_ledger.json',result)
    def table(headers,rows):
        return '<table><tr>'+''.join(f'<th>{html.escape(str(v))}</th>' for v in headers)+'</tr>'+''.join(
            '<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row)+'</tr>' for row in rows)+'</table>'
    pct=lambda x:'N/A' if x is None else f'{x:.2f}%'
    page='<meta charset="utf-8"><title>生成台账、AOR 与覆盖率</title><style>body{font:18px sans-serif;max-width:1400px;margin:30px auto}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px}</style><h1>车辆生产台账：AOR 与标注覆盖率</h1>'
    page+=f'<p>更新时间：{result["updated_utc"]}。只统计已提交批次，当前场景未结束时为阶段性结果。</p>'
    page+=table(['数据集','类别组','已完成批次查询','生成成功','导出且16视角有效','技术 AOR','保留','淘汰','失败','质量入库率'],
        [[r['dataset'],r['group'],r['completed_chunk_queries'],r['generated'],r['render_valid'],pct(r['AOR_percent']),
          r['counts'].get('keep',0),r['counts'].get('reject',0),r['counts'].get('failed',0),pct(r['bank_acceptance_percent'])] for r in summaries])
    page+='<p>AOR = 技术有效数 / 已提交批次查询数。外观淘汰和身份重复仍可技术有效；输入和渲染审核证据保留；部分淘汰 PLY 已按用户要求清理，技术 AOR 使用当时记录；生成前已做原生身份去重，未关联身份仍有不确定性。</p>'
    page+=table(['数据集/场景','标注身份分母','准入覆盖身份','保留覆盖身份','保留覆盖率','未可靠关联的保留资产','场景状态'],
        [[r['dataset']+'/'+r['scene_id'],r['denominator_annotated_vehicle_ids'],r['admitted_linked_ids'],r['kept_linked_ids'],
          pct(r['kept_coverage_percent']),r['kept_without_verified_annotation_link'],r['coverage_status']] for r in scenes if 'denominator_annotated_vehicle_ids' in r])
    page+='<p>覆盖率 = 有保留资产的已验证标注身份数 / 当前场景可投影的车辆标注身份数。分母在生成结果之前确定；未匹配资产不冒充已覆盖身份。nuScenes 包含遮挡 cuboid，Waymo 部分身份仍按相机分开；两种覆盖率不能直接横向排名，也不代表全部无标注物体的覆盖率。</p>'
    page+=table(['场景','检测框','轨迹片段','FLUX交接候选','去重后计划条目','生成前移除','源图JSON/API错误'],
        [[r['scene_id'],r['detections'],r['tracklets'],r['pre_identity_handoff_queries'],r['scheduled_queries'],r['pre_generation_removed'],r['source_schema_or_api_failed']] for r in scenes if 'detections' in r])
    page+='<p>轨迹片段不是物体个数。已冻结批次保留原输入，内部重复另行淘汰；计划条目不等于已证明唯一的物体数。格式/API错误单列，不当作真实图质量不合格。</p>'
    page+='<p><a href="production_ledger.json">全部查询及场景覆盖记录</a> · <a href="START_HERE.html">实时生成进度</a>。车辆与非车辆的外观指标在生产结束后分别计算。输入和当前新增过程文件保留；历史淘汰 PLY 清理见逐批清单。</p>'
    (out/'production_ledger.html').write_text(page)
    fields=['dataset','group','completed_chunk_queries','generated','render_valid','AOR_percent','bank_acceptance_percent','identity_rejections']
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(summaries)
    (out/'production_ledger.csv').write_text(buffer.getvalue())
    return result

if __name__=='__main__':
    if '--prepare-ground-truth' in sys.argv:
        print(json.dumps(prepare_ground_truth()))
    elif '--ledger' in sys.argv:
        result=production_ledger();print(json.dumps(result['summaries']))
    else:asyncio.run(main())
