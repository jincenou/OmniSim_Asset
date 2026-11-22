"""Frozen, kept-only production snapshots using the existing verified metric implementation."""
import asyncio,copy,json,os,sys,time,importlib.util
from pathlib import Path
from collections import defaultdict
from runtime import ROOT,LIBRARY,read_json,write_json,sha256,SharedGpuPool
from asset_extraction.vision import _bbox_iou
OLD=Path('/mnt/ojc/workplace1/asset_work/batch20_20261024')
SCRIPT=OLD/'scripts/evaluate_batch.py'
R=Path('/mnt/ojc/workplace1/reconviagen_demo')
SNAPSHOT=os.environ.get('VEHICLE_METRIC_SNAPSHOT','kept_first13')
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
    if REGISTRY.exists():return read_json(REGISTRY)
    entries=[r for r in read_json(LIBRARY/'manifests/asset_library_rigid.json')['assets'] if r['provenance'].get('batch_id')==ROOT.name and r['status']=='keep']
    rows=[];sources={};truth={}
    for entry in entries:
        d=entry['dataset'];scene=entry['provenance']['scene_id'];key=(d,scene)
        if key not in sources:
            run=ROOT/'runs'/f'{d}_{scene}'
            sources[key]={r['asset_id']:r for r in read_json(run/'scenes'/scene/'logs/export_items_full.json')}
            gt=ground_truth_path(scene)
            truth[key]=read_json(gt)['records'] if gt.exists() else []
        oid=entry['provenance']['source_obj_id'];source=sources[key][oid];obs=source['source']
        anchor=next(r for r in source['real_images'] if r['source_image']==obs['source_image'] and max(abs(a-b) for a,b in zip(r['bbox_xyxy'],obs['bbox_xyxy']))<.01)
        linked=sorted([(_bbox_iou(obs['bbox_xyxy'],t['bbox_xyxy']),i) for i,t in enumerate(truth[key]) if t['source_image']==obs['source_image']],reverse=True)
        native=truth[key][linked[0][1]] if linked and linked[0][0]>=.5 else None
        identity=entry['provenance'].get('production_identity',{}).get('native_id')
        if native and identity and native['id']!=identity:native=None
        eligible=bool(native and identity and native.get('pose_link') and entry['canonical']['heading_verified'])
        rows.append(dict(obj_id=oid,dataset=d,scene_id=scene,route=entry['route'],category=entry['category'],quality_status='keep',
            canonical=entry['canonical'],ply=str(LIBRARY/entry['canonical']['canonical_path']),ply_sha256=entry['canonical']['sha256'],
            source_rgba=anchor['rgba_path'],source_sha256=sha256(anchor['rgba_path']),source_crop_box=anchor['crop_box'],
            matched_annotation=native if eligible else None,matched_exclusion=None if eligible else 'Native object pose and verified heading unavailable for this anchor'))
    original=read_json(OLD/'frontend_revision_01/evaluation/metric_registry.json')['protocol'];protocol=copy.deepcopy(original)
    for item in protocol['checkpoints'].values():
        if sha256(item['path'])!=item['sha256']:raise ValueError('Metric checkpoint changed')
    protocol.update(version='vehicle_production_kept_snapshot_v1',script_sha256=sha256(SCRIPT),snapshot_script_sha256=sha256(__file__),
        queries=len(rows),exported_payloads=len(rows),partial_snapshot=True,created=time.time(),
        cohorts='Only final kept, deduplicated production assets at snapshot time. Rejected assets are excluded; this is not the all-valid-export benchmark.',
        failure_policy='Generation and curation ledgers retain all attempts separately. Metric errors remain explicit. Rejected outputs were deleted by user instruction; their similarity features are not reconstructed.',
        identity_protocol='Dataset-native identity metadata assists production deduplication, not RGB-only benchmark construction.',
        comparisons='nuScenes and Waymo contain different instances. These are cohort diagnostics, not controlled route comparisons.')
    protocol['admission_by_asset']={entry['provenance']['source_obj_id']:dict(
        policy=entry['quality'].get('admission_policy','historical_strict'),
        native_id=entry['provenance'].get('production_identity',{}).get('native_id'),
        retained_with_identity_variation=entry['quality'].get('retained_with_identity_variation',False))
        for entry in entries}
    result=dict(protocol=protocol,assets=rows);write_json(REGISTRY,result);return result

async def main():
    doc=freeze();out=ROOT/'evaluation';out.mkdir(exist_ok=True);(ROOT/'pilot_results').mkdir(exist_ok=True)
    devices=[int(value) for value in os.environ.get('VEHICLE_METRIC_GPUS','6').split(',')]
    if not devices or len(set(devices))!=len(devices):raise ValueError('Metric GPUs must be distinct')
    pool=SharedGpuPool(devices);evaluation_slot=asyncio.Semaphore(1)
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
    await launch('/home/wmj/miniconda3/envs/sim/bin/python',['render','--indices',*map(str,range(len(doc['assets'])))],f'{SNAPSHOT}_render')
    feature_python='/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python'
    await asyncio.gather(*(launch(feature_python,['features','--encoder',encoder],f'{SNAPSHOT}_{encoder}',stage='H' if encoder=='dino' else 'render') for encoder in ['clip','dino','fid']),
        launch(str(R/'env_refine/bin/python'),['pixels'],f'{SNAPSHOT}_pixels'))
    await launch(feature_python,['summarize'],f'{SNAPSHOT}_summary',gpu=False)
    path=ROOT/'pilot_results'/f'{SNAPSHOT}.json';result=read_json(path)
    result['summary']=[row for row in result['summary'] if row['cohort']=='final_keep']
    validate_matched_summary(result)
    write_json(path,result)
    import csv
    fields=['dataset','n_kept','CLIP_I16','DINO_I16','ED_R','PSNR_dB','SSIM','LPIPS_alex','silhouette_IoU','FID_2048']
    with (ROOT/'pilot_results'/f'{SNAPSHOT}_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in result['summary']:
            writer.writerow(dict(dataset=row['dataset'],n_kept=row['valid_queries'],**{key:row['metrics'][key].get('mean',row['metrics'][key].get('value')) for key in fields[2:]}))
    publish_metrics(result)
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
        rows=[r for r in result['assets'] if r['dataset']==summary['dataset'] and r['valid_16_views']]
        for key in keys:
            values=[r[key] for r in rows if r.get(key) is not None]
            summary['metrics'][key]=dict(n=len(values),mean=sum(values)/len(values) if values else None)
    result['protocol']['matched_pose_exclusions']=exclusions

def publish_metrics(result):
    import html,shutil
    from collections import Counter
    out=LIBRARY/'previews'/ROOT.name
    write_json(out/f'{SNAPSHOT}.json',result)
    shutil.copyfile(ROOT/'pilot_results'/f'{SNAPSHOT}_summary.csv',out/f'{SNAPSHOT}.csv')
    policy=result['protocol'].get('admission_by_asset',{})
    counts=dict(Counter(v['policy'] for v in policy.values()))
    variations=sum(v.get('retained_with_identity_variation',False) for v in policy.values())
    count=len(result['assets'])
    page=f'<meta charset="utf-8"><title>{count}个保留资产指标</title><style>body{{font:18px sans-serif;max-width:1450px;margin:30px auto}}table{{border-collapse:collapse}}td,th{{border:1px solid #aaa;padding:8px}}</style><h1>{count} 个保留车辆的固定指标快照</h1>'
    page+='<p>包含旧严格准入和允许身份差异的新规则。'+html.escape(str(counts))+f'；其中 {variations} 个标记为身份差异保留。不同数据集/规则组的对象不同，数值不能作为算法或规则优劣的因果对比。</p>'
    page+='<table><tr><th>数据集</th><th>有效对象</th><th>CLIP-I16 ↑</th><th>DINO-I16 ↑</th><th>FID ↓</th><th>ED-R ↓</th><th>PSNR ↑</th><th>SSIM ↑</th><th>LPIPS ↓</th><th>IoU ↑</th></tr>'
    for row in result['summary']:
        cells=[row['dataset'],str(row['valid_queries'])]
        for key in ['CLIP_I16','DINO_I16','FID_2048','ED_R','PSNR_dB','SSIM','LPIPS_alex','silhouette_IoU']:
            m=row['metrics'][key];v=m.get('mean',m.get('value'));n=m.get('n',m.get('n_identities'))
            cells.append('N/A' if v is None else f'{v:.3f} (n={n})')
        page+='<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in cells)+'</tr>'
    page+='</table><p>源图是数据集原始 SAM 前景，未用 FLUX 图充当真实图。CLIP ViT-B/32 与 DINOv3 ViT-H/16+ 按对象平均16个固定环绕视角；FID 使用 Inception-2048，源图与16视角渲染分别组成分布。前景5%留白、白底512像素，再使用各编码器的预处理。</p>'
    page+='<p>ED-R、PSNR、SSIM、LPIPS 和 IoU 只在原生物体姿态与资产正向可靠时计算，使用固定源相机和各向同性前景归一化，不搜索最佳视角。无可靠配准者为 N/A；统计单元是物体。输入图参与了构建，因此是源条件一致性而不是独立测试图结果。</p>'
    page+='<p>此表仅覆盖快照时已保留的资产，不含已删除淘汰结果。样本数量有限、渲染视角彼此相关；FID 不能代表场景插入效果。AOR 和覆盖率另按完整生产台账计算。</p>'
    page+=f'<p><a href="{SNAPSHOT}.csv">CSV表格</a> · <a href="{SNAPSHOT}.json">逐对象结果及模型哈希</a> · <a href="production_ledger.html">AOR和覆盖率</a> · <a href="START_HERE.html">生成进度</a></p>'
    (out/f'{SNAPSHOT}.html').write_text(page)
    write_json(out/'metrics_latest.json',dict(snapshot=SNAPSHOT,assets=count,html=f'{SNAPSHOT}.html',updated=time.time()))

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
        generation={r['obj_id']:r for r in read_json(chunk/'generation/generation.json')['assets']}
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
                chunk=chunk.name,generated=gen['status']=='generated',valid_at_construction=renderable,
                status=audit['status'],visual_status=audit.get('visual_curation_status',audit['status']),
                identity_rejection=audit.get('identity_rejection'),error=audit.get('error'),
                evidence_sha256=render.get('ply_sha256'),rendered_views=len(pixels),
                rejected_outputs_deleted=(chunk/'rejection_archive.json').exists() and audit['status']=='reject'))
    summaries=[]
    for dataset in ['nuscenes','waymo']:
        rows=[r for r in attempts if r['dataset']==dataset];n=len(rows)
        summaries.append(dict(dataset=dataset,completed_chunk_queries=n,generated=sum(r['generated'] for r in rows),
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
        kept=[r for r in bank if r['dataset']==dataset and r['provenance']['scene_id']==scene and r['status']=='keep']
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
    definitions=dict(
        AOR='At construction time: queries with a decoded nonempty Gaussian and 16 nonempty orbit views / all queries in committed chunks, including generation or render failures. Visual quality rejection and duplicate rejection do not erase technical validity.',
        AOR_scope='Operational query-attempt rate. Older frozen chunk prefixes contain repeat identities, so this is not the paper single-query-per-physical-identity benchmark. Pending chunks are excluded from this snapshot, not counted as failures.',
        coverage='Unique verified native vehicle IDs with a kept asset / fixed native annotated vehicle IDs in the processed scene frames. Unmatched kept assets do not increase this numerator.',
        coverage_limits='nuScenes uses in-frustum cuboids, including occluded objects; Waymo uses camera annotations and only verified cross-camera links. Different annotation scopes make the two percentages non-comparable as algorithm rankings. Not coverage of all unannotated scene objects.',
        source_failures='Schema/API errors are distinct from source-quality rejection. Historical failures remain recorded; later frontend fixes do not rewrite them.',
        missing_metrics='FID and other similarity metrics are on the separate frozen kept-only subset, not these entire operational cohorts. FAD remains undefined and is not computed.')
    old_registry=read_json(OLD/'registry.json');pilot=OLD/'frontend_revision_01/reports'
    scope=[dict(dataset=d,scene_id=s,mixed_categories_complete=(s==first and (pilot/f'{d}_generation_complete.json').exists()),
                vehicle_production_status=read_json(ROOT/f'{d}_status.json')['scenes'].get(s,{}).get('status','not_started'))
           for d,first,items in [('nuscenes','scene-0001',old_registry['nuscenes']),
                 ('waymo','waymo_val_000',[r['scene_id'] for r in old_registry['waymo']])]
           for s in items]
    result=dict(updated_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),summaries=summaries,
        scenes=scenes,attempts=attempts,definitions=definitions,first20_scope=scope)
    out=LIBRARY/'previews'/ROOT.name
    write_json(ROOT/'evaluation/production_ledger.json',result);write_json(out/'production_ledger.json',result)
    def table(headers,rows):
        return '<table><tr>'+''.join(f'<th>{html.escape(str(v))}</th>' for v in headers)+'</tr>'+''.join(
            '<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row)+'</tr>' for row in rows)+'</table>'
    pct=lambda x:'N/A' if x is None else f'{x:.2f}%'
    page='<meta charset="utf-8"><title>生成台账、AOR 与覆盖率</title><style>body{font:18px sans-serif;max-width:1400px;margin:30px auto}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px}</style><h1>车辆生产台账：AOR 与标注覆盖率</h1>'
    page+=f'<p>更新时间：{result["updated_utc"]}。只统计已提交批次，当前场景未结束时为阶段性结果。</p>'
    page+=table(['数据集','已完成批次查询','生成成功','导出且16视角有效','技术 AOR','保留','淘汰','失败','质量入库率'],
        [[r['dataset'],r['completed_chunk_queries'],r['generated'],r['render_valid'],pct(r['AOR_percent']),
          r['counts'].get('keep',0),r['counts'].get('reject',0),r['counts'].get('failed',0),pct(r['bank_acceptance_percent'])] for r in summaries])
    page+='<p>AOR = 技术有效数 / 已提交批次查询数。外观淘汰和身份重复仍可技术有效；淘汰输出删除前的渲染和校验记录保留。旧冻结批次中存在同一物体的多次尝试，因此这里是生产查询口径，不能直接作为论文的独立物体 AOR。</p>'
    page+=table(['数据集/场景','标注身份分母','准入覆盖身份','保留覆盖身份','保留覆盖率','未可靠关联的保留资产','场景状态'],
        [[r['dataset']+'/'+r['scene_id'],r['denominator_annotated_vehicle_ids'],r['admitted_linked_ids'],r['kept_linked_ids'],
          pct(r['kept_coverage_percent']),r['kept_without_verified_annotation_link'],r['coverage_status']] for r in scenes if 'denominator_annotated_vehicle_ids' in r])
    page+='<p>覆盖率 = 有保留资产的已验证标注身份数 / 当前场景可投影的车辆标注身份数。分母在生成结果之前确定；未匹配资产不冒充已覆盖身份。nuScenes 包含遮挡 cuboid，Waymo 部分身份仍按相机分开；两种覆盖率不能直接横向排名，也不代表全部无标注物体的覆盖率。</p>'
    page+=table(['场景','检测框','轨迹片段','FLUX交接候选','去重后计划条目','生成前移除','源图JSON/API错误'],
        [[r['scene_id'],r['detections'],r['tracklets'],r['pre_identity_handoff_queries'],r['scheduled_queries'],r['pre_generation_removed'],r['source_schema_or_api_failed']] for r in scenes if 'detections' in r])
    page+='<p>轨迹片段不是物体个数。已冻结批次保留原输入，内部重复另行淘汰；计划条目不等于已证明唯一的物体数。格式/API错误单列，不当作真实图质量不合格。</p>'
    page+=f'<p>原各10场景混合类别任务：仅 {sum(r["mixed_categories_complete"] for r in scope)} / 20 场景完整完成；其余场景不能用千车任务代替。<a href="production_ledger.json">完整统计、逐查询证据和20场景清单</a> · <a href="metrics_first12.html">独立12对象指标快照</a> · <a href="START_HERE.html">实时生成进度</a></p>'
    (out/'production_ledger.html').write_text(page)
    fields=['dataset','completed_chunk_queries','generated','render_valid','AOR_percent','bank_acceptance_percent','identity_rejections']
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(summaries)
    (out/'production_ledger.csv').write_text(buffer.getvalue())
    return result

if __name__=='__main__':
    if '--prepare-ground-truth' in sys.argv:
        print(json.dumps(prepare_ground_truth()))
    elif '--ledger' in sys.argv:
        result=production_ledger();print(json.dumps(result['summaries']))
    else:asyncio.run(main())
