"""Scene-ordered, independent 500-keep quotas for nuScenes and Waymo vehicles."""
import argparse,asyncio,fcntl,importlib.util,json,os,sys,time,html,shutil,signal
from pathlib import Path
import yaml
from runtime import ROOT,P,LIBRARY,GPUS,SharedGpuPool,process_chunk,read_json,write_json,sha256
from asset_extraction.config import load_config
from asset_extraction.real_views import VEHICLES
from asset_extraction.vision import _bbox_iou
PYTHON='/mnt/ojc/miniconda3/envs/vllm_vlm/bin/python'
OLD=Path('/mnt/ojc/workplace1/asset_work/batch20_20261024')
REV=OLD/'frontend_revision_01'
TARGET=500;CHUNK=24
FRONTEND_TASKS={}
DRAIN=False
os.environ['VLM_BASE_URL']='http://127.0.0.1:8082/v1';os.environ['VLM_RESPONSE_FORMAT']='json_object'
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'

def registry():
    path=ROOT/'registry.json'
    if path.exists():return read_json(path)
    waymo=[dict(scene_id=f'waymo_val_{i:03d}',raw_tfrecord=str(p)) for i,p in enumerate(sorted(Path('/mnt/public_data/waymo_raw/validation').glob('*.tfrecord')))]
    nu=sorted(read_json(Path('/mnt/public_data/nuScenes/v1.0-trainval/scene.json')),key=lambda r:r['name'])
    result=dict(nuscenes=[dict(scene_id=r['name']) for r in nu],waymo=waymo,
        target_new_keep_per_dataset=TARGET,order='scene name / TFRecord filename ascending',
        existing_complete_pilot_scenes={'nuscenes':['scene-0001'],'waymo':['waymo_val_000']},
        curation='original strict keep/review/reject; only keep counts toward quota',
        rejection_policy='archive JSON decision and hashes, delete new rejected generated outputs',
        created=time.time())
    write_json(path,result);return result

def prepare_nuscenes():
    from asset_extraction.pipeline import _prepare_nuscenes_scene_dirs
    config=load_config(REV/'configs/nuscenes_pilot.yaml');config.scene_ids=[];config.limit_scenes=None
    paths=_prepare_nuscenes_scene_dirs(config,ROOT/'inputs/nuscenes')
    write_json(ROOT/'inputs/nuscenes/complete.json',dict(scenes=[p.stem for p in paths],completed=time.time()))

def prepare_waymo(scene):
    spec=importlib.util.spec_from_file_location('prepare_waymo_native',OLD/'scripts/prepare_waymo.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.ROOT=ROOT;module.worker(scene)

async def command(args,log,env=None,on_poll=None):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('ab') as stream:
        proc=await asyncio.create_subprocess_exec(*map(str,args),cwd=P,env=env or os.environ.copy(),stdout=stream,stderr=asyncio.subprocess.STDOUT,start_new_session=True)
        waiter=asyncio.create_task(proc.wait())
        try:
            while not waiter.done():
                await asyncio.wait({waiter},timeout=2)
                if on_poll is not None:on_poll()
            code=waiter.result()
        finally:
            if proc.returncode is None:
                os.killpg(proc.pid,signal.SIGTERM)
                await waiter
    if code:raise RuntimeError(f'Command exited {code}; log={log}')

def source_observations(source):
    return [source,*source.get('frame_records',[])]

def prior_observations(dataset):
    known={}
    for row in read_json(LIBRARY/'manifests/asset_library_rigid.json')['assets']:
        if row['dataset']!=dataset or row['category'] not in VEHICLES:continue
        source=row['provenance']['source']
        for obs in source_observations(source):
            camera=obs.get('camera_name',source.get('camera_name'))
            if obs.get('timestamp_us') is not None:
                key=(camera,round(obs['timestamp_us']/10000))
                known.setdefault(key,[]).append((row['uid'],obs['bbox_xyxy']))
    return known

def remove_existing(rows,routes,dataset):
    known=prior_observations(dataset);kept=[];decisions=[]
    byid={r['obj_id']:r for r in routes}
    for row in rows:
        source=byid[row['obj_id']]['source'];match=None
        for obs in source_observations(source):
            if obs.get('timestamp_us') is None:continue
            key=(obs.get('camera_name',source.get('camera_name')),round(obs['timestamp_us']/10000))
            for uid,bbox in known.get(key,[]):
                if _bbox_iou(obs['bbox_xyxy'],bbox)>=.8:match=uid;break
            if match:break
        if match:decisions.append(dict(obj_id=row['obj_id'],duplicate_of=match,method='same camera and timestamp; bbox IoU >= 0.8'))
        else:kept.append(row)
    return kept,decisions

def frontend_task(dataset,item):
    key=(dataset,item['scene_id']);task=FRONTEND_TASKS.get(key)
    if task is None or task.done() and (task.cancelled() or task.exception() is not None):
        task=asyncio.create_task(_frontend(dataset,item),name='frontend:'+':'.join(key))
        FRONTEND_TASKS[key]=task
        def finished(done):
            if not done.cancelled() and done.exception() is not None:
                print('Frontend preparation failed:',key,repr(done.exception()),flush=True)
        task.add_done_callback(finished)
    return task

async def frontend(dataset,item):
    return await asyncio.shield(frontend_task(dataset,item))

def prefetch_next(dataset,scene,items,skip,available,state):
    if DRAIN or TARGET-state['kept']<=CHUNK:return
    remaining=[item for item in items if item['scene_id']>scene and item['scene_id'] not in skip
               and item['scene_id'] in available and state['scenes'].get(item['scene_id'],{}).get('status')!='completed']
    if remaining:
        item=remaining[0]
        frontend_task(dataset,item)
        write_json(ROOT/f'{dataset}_prefetch.json',dict(scene_id=item['scene_id'],status='scheduled',updated=time.time()))

async def _frontend(dataset,item):
    scene=item['scene_id'];name=f'{dataset}_{scene}';run=ROOT/'runs'/name
    config_path=ROOT/'configs'/f'{name}.yaml'
    if (run/'handoff/asset_routes.json').exists():return run,load_config(config_path)
    if dataset=='waymo':
        old_index=OLD/'inputs/waymo_indexes'/f'{scene}.json';index=ROOT/'inputs/waymo_indexes'/f'{scene}.json'
        if old_index.exists():
            index.parent.mkdir(parents=True,exist_ok=True)
            if not index.exists():index.symlink_to(old_index)
        elif not index.exists():
            await command([PYTHON,__file__,'--prepare-waymo',scene],ROOT/'logs'/f'prepare_{scene}.log')
        index_root=index.parent;cameras=['front','front_left','front_right','side_left','side_right']
    else:
        index_root=ROOT/'inputs/nuscenes/debug/dataset_scenes'
        cameras=['front','front_left','front_right','back','back_left','back_right']
    cfg=yaml.safe_load((REV/'configs'/f'{dataset}_pilot.yaml').read_text())
    cfg.update(dataset='frames',dataset_root=str(index_root),scene_ids=[scene],limit_scenes=1,
        output_root=str(ROOT),run_name=name,camera_channels=cameras,classes=['vehicle','car','truck','bus','van'],
        detection_cache_root=None,gpu_ids=GPUS)
    cfg['vlm']['endpoint']='http://127.0.0.1:8082/v1'
    config_path.parent.mkdir(parents=True,exist_ok=True);content=yaml.safe_dump(cfg,sort_keys=False)
    if config_path.exists() and config_path.read_text()!=content:raise ValueError('Frontend config changed')
    config_path.write_text(content)
    # Keep bus GPU 0/2 available for its priority lane while its run is active.
    bus_done=Path('/mnt/ojc/workplace1/asset_work/waymo_bus_more_20261025/completed.json').exists()
    pool=SharedGpuPool(GPUS if bus_done else [1,3,4,5,7]);gpu=await pool.acquire('frontend')
    env=dict(os.environ,CONFIG=str(config_path),PYTHON=PYTHON,GPUS=str(gpu),RUN_NAME=name,OUTPUT_ROOT=str(ROOT))
    status_path=run/'scenes'/scene/'status.json'
    initial_status=status_path.stat().st_mtime_ns if status_path.exists() else None
    released=False
    def release_after_extraction():
        nonlocal released
        if released or not status_path.exists() or status_path.stat().st_mtime_ns==initial_status:return
        data=read_json(status_path)
        if data.get('stage')=='scene_image_export' and data.get('status')=='completed':
            # The scene worker has returned from detection/SAM and now only
            # sends async VLM requests. The allocator still checks free memory.
            pool.release(gpu);released=True
            write_json(run/'reports/gpu_release.json',dict(gpu=gpu,stage='source_vlm',released=time.time(),
                reason='Detection and segmentation completed; subsequent stages use the shared VLM service'))
            print(f'Frontend GPU released before VLM: {name} gpu={gpu}',flush=True)
    try:await command(['bash','scripts/run_nuscenes.sh','--handoff-only'],ROOT/'logs'/f'{name}_frontend.log',env,release_after_extraction)
    finally:
        if not released:pool.release(gpu)
    write_json(ROOT/f'{dataset}_prefetch.json',dict(scene_id=scene,status='ready',updated=time.time()))
    return run,load_config(config_path)

async def dataset_run(dataset):
    state_path=ROOT/f'{dataset}_status.json'
    state=read_json(state_path) if state_path.exists() else dict(dataset=dataset,target_keep=TARGET,kept=0,queries=0,scenes={},chunks=[],status='starting')
    def save(**kwargs):state.update(kwargs,updated=time.time());write_json(state_path,state)
    reg=registry();skip=set(reg['existing_complete_pilot_scenes'][dataset])
    if dataset=='nuscenes':
        while not (ROOT/'inputs/nuscenes/complete.json').exists():
            save(status='indexing_nuscenes');await asyncio.sleep(10)
        available=set(read_json(ROOT/'inputs/nuscenes/complete.json')['scenes'])
    else:available={r['scene_id'] for r in reg[dataset]}
    for item in reg[dataset]:
        scene=item['scene_id']
        if state['kept']>=TARGET or DRAIN:break
        if scene in skip or scene not in available or state['scenes'].get(scene,{}).get('status')=='completed':continue
        save(status='frontend',current_scene=scene)
        try:
            run,config=await frontend(dataset,item)
            if DRAIN:break
            prefetch_next(dataset,scene,reg[dataset],skip,available,state)
            selection=run/'vehicle_selection.json';routing=read_json(run/'handoff/asset_routes.json')
            all_rows=read_json(run/'handoff/flux_input.json')
            if selection.exists():selected=read_json(selection)
            else:
                rows=[r for r in all_rows if r['category'] in VEHICLES]
                rows,duplicates=remove_existing(rows,routing['assets'],dataset)
                selected=dict(rows=rows,duplicates=duplicates,all_frontend_admitted=len(all_rows));write_json(selection,selected)
            rows=selected['rows'];routes={r['obj_id']:r for r in routing['assets']}
            identity_path=ROOT/'identity'/f'{dataset}_{scene}.json'
            if not identity_path.exists():
                await command([PYTHON,ROOT/'identity.py',dataset,scene,run],ROOT/'logs'/f'{dataset}_{scene}_identity.log')
            production=run/'production_selection.json'
            if production.exists():
                unique=read_json(production)
            else:
                from identity import distinct_selection
                # Resume frozen in-flight chunks exactly, then deduplicate all unscheduled candidates.
                existing=list((ROOT/'chunks').glob(f'{dataset}_{scene}_*/flux_input.json'))
                prefix=max((int(p.parent.name.rsplit('_',1)[1])+len(read_json(p)) for p in existing),default=0)
                reserved=rows[:prefix]
                identity=read_json(identity_path)
                tail,tail_routes,removed=distinct_selection(rows[prefix:],routing['assets'],identity['matches'],[r['obj_id'] for r in reserved],
                    [r['native_id'] for r in identity.get('existing_assets',{}).values()])
                unique=dict(rows=reserved+tail,routes=[routes[r['obj_id']] for r in reserved]+tail_routes,
                    scheduled_prefix=prefix,removed=removed,protocol='native identity assisted production; sensor geometry not supplied to reconstruction')
                write_json(production,unique)
            rows=unique['rows'];routes={r['obj_id']:r for r in unique['routes']}
            progress=state['scenes'].setdefault(scene,dict(offset=0,status='processing',vehicle_queries=len(rows),duplicate_queries=len(selected['duplicates'])))
            progress.update(vehicle_queries=len(rows),native_duplicate_queries=len(unique['removed']),identity_protocol=unique['protocol'])
            while progress['offset']<len(rows) and state['kept']<TARGET and not DRAIN:
                start=progress['offset'];amount=min(CHUNK,TARGET-state['kept'],len(rows)-start)
                subset=rows[start:start+amount];chunk=ROOT/'chunks'/f'{dataset}_{scene}_{start:04d}'
                save(status='generation_and_curation',current_chunk=str(chunk))
                subroutes=dict(routing,assets=[routes[r['obj_id']] for r in subset])
                done=await process_chunk(chunk,subset,subroutes,config,dataset,ROOT.name)
                if str(chunk) not in state['chunks']:
                    state['kept']+=done['kept'];state['queries']+=done['queries'];state['chunks'].append(str(chunk))
                progress['offset']+=amount;save()
            if progress['offset']==len(rows):progress['status']='completed'
            save()
        except Exception as exc:
            save(status='error',error=f'{type(exc).__name__}: {exc}');raise
    save(status='target_reached' if state['kept']>=TARGET else 'drained' if DRAIN else 'available_scenes_exhausted')

async def main():
    ROOT.mkdir(parents=True,exist_ok=True);registry()
    def drain():
        global DRAIN
        DRAIN=True
        print('Drain requested: finish active chunks and prepared frontends before exiting.',flush=True)
    asyncio.get_running_loop().add_signal_handler(signal.SIGUSR1,drain)
    with (ROOT/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        async def indexing():
            if not (ROOT/'inputs/nuscenes/complete.json').exists():
                await command([PYTHON,__file__,'--prepare-nuscenes'],ROOT/'logs/index_nuscenes.log')
        async def supervise(dataset):
            while not DRAIN:
                try:await dataset_run(dataset);return
                except Exception as exc:
                    print(dataset,'explicit error; retry in 60 seconds:',repr(exc),flush=True)
                    await asyncio.sleep(60)
        try:await asyncio.gather(indexing(),supervise('nuscenes'),supervise('waymo'))
        finally:
            if DRAIN:
                await asyncio.gather(*FRONTEND_TASKS.values(),return_exceptions=True)
            else:
                for task in FRONTEND_TASKS.values():
                    if not task.done():task.cancel()
                await asyncio.gather(*FRONTEND_TASKS.values(),return_exceptions=True)

async def maintenance():
    """Publish compact previews and progress for live vehicle workers."""
    from PIL import Image,ImageOps,ImageDraw,ImageFont
    out=LIBRARY/'previews'/ROOT.name;out.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype('/mnt/ojc/miniconda3/envs/vllm_vlm/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf',18)
    with (ROOT/'maintenance.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger_signature=None
        while True:
            states={d:read_json(ROOT/f'{d}_status.json') for d in ('nuscenes','waymo') if (ROOT/f'{d}_status.json').exists()}
            chunks=list((ROOT/'chunks').glob('*/completed.json'))
            entries=[r for r in read_json(LIBRARY/'manifests/asset_library_rigid.json')['assets'] if r['provenance'].get('batch_id')==ROOT.name]
            current_signature=(tuple(sorted((str(p),p.stat().st_mtime_ns) for p in chunks)),len(entries),
                tuple(sorted(str(p) for p in (ROOT/'runs').glob('*/production_selection.json'))))
            if current_signature!=ledger_signature:
                try:
                    from metrics import production_ledger
                    production_ledger()
                    ledger_signature=current_signature
                    write_json(ROOT/'ledger_status.json',dict(status='updated',updated=time.time()))
                except Exception as exc:
                    write_json(ROOT/'ledger_status.json',dict(status='error',error=f'{type(exc).__name__}: {exc}',updated=time.time()))
                    print('Statistics update failed:',repr(exc),flush=True)
            metrics={d:dict(generated=0,curated=0,keep=0,review=0,reject=0,failed=0) for d in states}
            for path in (ROOT/'chunks').glob('*/generation/generation.json'):
                dataset='nuscenes' if path.parent.parent.name.startswith('nuscenes_') else 'waymo'
                if dataset in metrics:metrics[dataset]['generated']+=sum(r['status']=='generated' for r in read_json(path)['assets'])
            for path in (ROOT/'chunks').glob('*/curation/curation.json'):
                dataset='nuscenes' if path.parent.parent.name.startswith('nuscenes_') else 'waymo'
                if dataset not in metrics:continue
                for result in read_json(path)['assets'].values():
                    metrics[dataset]['curated']+=1;metrics[dataset][result['status']]+=1
            cards=[]
            for row in sorted(entries,key=lambda r:(r['dataset'],r['uid'])):
                if row['status']!='keep':continue
                name=row['canonical']['sha256'][:20]+'.jpg';target=out/name
                if not target.exists():
                    evidence=Image.open(row['quality']['rendering']['evidence']);s=row['quality']['rendering']['resolution']
                    images=[Image.open(row['provenance']['source_image'])]
                    for index in (2,10):
                        x=index%4*s;y=index//4*(s+24)+24;images.append(evidence.crop((x,y,x+s,y+s)))
                    sheet=Image.new('RGB',(768,285),'#172331');draw=ImageDraw.Draw(sheet)
                    for index,im in enumerate(images):
                        draw.text((256*index+6,6),('SOURCE','45 deg','225 deg')[index],font=font,fill='white')
                        sheet.paste(ImageOps.pad(im.convert('RGB'),(250,250),color='white' if index==0 else '#808080'),(256*index+3,32))
                    sheet.save(target,quality=88)
                label=html.escape(row['provenance']['source_obj_id'])+(' · 身份差异保留' if row['quality'].get('retained_with_identity_variation') else '');ply=os.path.relpath(LIBRARY/row['canonical']['canonical_path'],out)
                cards.append(f'<article><p>{row["dataset"]} / {label} / {row["category"]}</p><a href="{ply}">PLY</a><br><img loading="lazy" src="{name}"></article>')
            snapshot=dict(updated=time.time(),states=states,live_counts=metrics,published_kept=len(cards),
                note='Generated counts are not quality-qualified counts. Only keep reaches the per-dataset quota. FID and coverage are not inferred from these operational counts.')
            write_json(ROOT/'snapshot.json',snapshot);write_json(out/'progress.json',snapshot)
            page='<meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>车辆批量生成进度</title><style>body{font:18px sans-serif;background:#172331;color:#eee;margin:30px}a{color:#9dd7ff}table{border-collapse:collapse}td,th{border:1px solid #567;padding:8px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:16px}img{max-width:100%}</style><h1>车辆批量生成 · 每个数据集 500 个合格资产</h1>'
            page+=f'<p>更新时间（北京时间）：{time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime(time.time()+8*3600))}。仅 keep 计入目标；review 和 reject 不计入。页面每 60 秒刷新。</p><table><tr><th>数据集</th><th>生成完成</th><th>保留</th><th>待复核</th><th>淘汰</th><th>当前阶段</th></tr>'
            for dataset,state in states.items():
                m=metrics[dataset];page+=f'<tr><td>{dataset}</td><td>{m["generated"]}</td><td>{state["kept"]} / 500</td><td>{m["review"]}</td><td>{m["reject"]}</td><td>{html.escape(state["status"])} / {html.escape(state.get("current_scene",""))}</td></tr>'
            page+='</table><p>后续车辆整体质量通过时允许保留身份差异，并在卡片标注；历史审核结果未重算。</p><p><a href="../bus/waymo_bus_more_20261025/START_HERE.html">优先交付的 Waymo 客车</a></p>'
            if (out/'metrics_latest.json').exists():
                latest=read_json(out/'metrics_latest.json')
                page+=f'<p><a href="{html.escape(latest["html"],quote=True)}">最新 {latest["assets"]} 个保留车辆的指标、FID 和评估口径</a></p>'
            elif (out/'metrics_first12.html').exists():page+='<p><a href="metrics_first12.html">首批 12 个独立保留车辆的指标、FID 和评估口径</a></p>'
            if (out/'production_ledger.html').exists():page+='<p><a href="production_ledger.html">完整生产台账、AOR 与标注覆盖率</a></p>'
            if (out/'rejection_reasons.html').exists():page+='<p><a href="rejection_reasons.html">旧规则下277个淘汰的原因审计</a></p>'
            page+='<div class="cards">'+''.join(cards)+'</div>'
            temp=out/'START_HERE.html.tmp';temp.write_text(page);temp.replace(out/'START_HERE.html')
            write_json(ROOT/'maintenance_status.json',dict(status='running',updated=time.time(),completed_chunks=len(chunks),published_kept=len(cards)))
            if len(states)==2 and all(s['status'] in {'target_reached','available_scenes_exhausted'} for s in states.values()):return
            await asyncio.sleep(20)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare-nuscenes',action='store_true');p.add_argument('--prepare-waymo');p.add_argument('--maintain',action='store_true');args=p.parse_args()
    if args.prepare_nuscenes:prepare_nuscenes()
    elif args.prepare_waymo:prepare_waymo(args.prepare_waymo)
    elif args.maintain:asyncio.run(maintenance())
    else:asyncio.run(main())
