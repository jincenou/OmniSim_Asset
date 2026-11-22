"""Resumable, bounded H production of reviewed complete seated rider assemblies."""
import asyncio,collections,fcntl,json,os,shutil,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'scene_production_20261031'))
import runtime
from asset_extraction import generation as gen
from asset_extraction.config import PipelineConfig,VLMConfig
from asset_extraction.vlm import AsyncOpenAIClient
from asset_extraction.io_utils import read_json,write_json
from asset_extraction.rider_vehicle import prompt,assembly
from asset_extraction.export import curate,_verdict
from asset_extraction.bank import publish
from review import ask,pair,SOURCE,PAIR,MASK,SUPPORT,BOX,schema
from PIL import Image
os.environ.update(VLM_BASE_URL='http://127.0.0.1:8082/v1',VLM_RESPONSE_FORMAT='json_object',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
GPUS=[0,1,2,3,5,6]
PYTHON='/mnt/ojc/miniconda3/envs/vllm_vlm/bin/python'
INPUT=ROOT.parent/'openimages_vehicle_wave_20261105'
CONFIG=PipelineConfig(mode='offline',workspace=Path('/mnt/ojc/DriveLab'),output_root=ROOT,run_name=ROOT.name,dataset='web',vlm=VLMConfig(endpoint='http://127.0.0.1:8082/v1',concurrency=4))
POOL=runtime.SharedGpuPool(GPUS)
VLM_LIMIT=asyncio.Semaphore(4)
LOCATE=schema(rider_box=BOX,vehicle_box=BOX)

def status(stage,**kw):
    value=dict(status=stage,pid=os.getpid(),updated=time.time(),target_keep=100,**kw)
    write_json(ROOT/'status.json',value);print(json.dumps(value),flush=True)

async def check(client,prompt_,image,definition,path):
    async with VLM_LIMIT:return await ask(client,prompt_,Path(image),definition,Path(path))

async def segment(items,label):
    if not items:return
    packet=write_json(ROOT/'packets'/f'{label}.json',items)
    gpu=await POOL.acquire('frontend')
    try:
        with (ROOT/'logs'/f'{label}.log').open('ab') as log:
            proc=await asyncio.create_subprocess_exec(PYTHON,str(ROOT/'segment.py'),str(packet),env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=asyncio.subprocess.STDOUT)
            if await proc.wait():raise RuntimeError('SAM worker failed: '+str(packet))
    finally:POOL.release(gpu)

async def source_one(client,a):
    uid='oi_'+a['split']+'_'+a['image_id'];folder=ROOT/'sources'/uid;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'done.json').exists():return None
    try:
        write_json(folder/'source.json',a)
        crop=folder/'input.jpg'
        if not crop.exists():shutil.copyfile(a['approved_input'],crop)
        evidence=read_json(a['approved_assembly_review'])
        if evidence['image_sha256']!=gen.sha256(crop):raise ValueError('Approved source image changed')
        d=evidence['decision'];write_json(folder/'source_review.json',evidence)
        write_json(folder/'seated_review.json',read_json(a['approved_seated_review']))
        accepted=all(d[k] for k in ('accepted','complete_assembly','exactly_one_rider','seated')) and (d['category']=='bicycle' or d['category']=='motorcycle' and d['propulsion_label']=='electric_probable')
        if not accepted:
            write_json(folder/'done.json',dict(status='source_rejected',reason=d['reason'],category=d['category'],propulsion=d['propulsion_label']));return None
        return dict(uid=uid,folder=folder,source=a,decision=d,image=crop)
    except Exception as exc:
        write_json(folder/'done.json',dict(status='source_error',error=repr(exc)));return None

async def prepare(client,items,label):
    reviewed=[a for a in await asyncio.gather(*(source_one(client,a) for a in items)) if a]
    await segment([dict(image=str(a['image']),output=str(a['folder']/'segmentation'),rider_box=a['decision']['rider_box'],vehicle_box=a['decision']['vehicle_box']) for a in reviewed],label)
    async def finish(a):
        folder=a['folder'];d=a['decision'];uid=a['uid']
        try:
            s=read_json(folder/'segmentation/result.json')
            if s['status']!='segmented':raise ValueError(s)
            review_image=pair(a['image'],s['anchor'],folder/'mask_comparison.jpg')
            m=await check(client,'Compare A original and B foreground cutout. Is the entire selected seated rider AND vehicle preserved including both wheels, head, main limbs, seat and handlebars? Small spoke gaps and minor boundary artifacts are acceptable. Reject person-only or bike-only cutouts, substantial missing wheel/body regions, or substantial unrelated background.',review_image,MASK,folder/'mask_review.json')
            if not all(m[k] for k in MASK['properties'] if k!='reason'):raise ValueError('Incomplete union mask: '+m['reason'])
            combo={k:d[k] for k in ['vehicle_subtype','propulsion_label','propulsion_evidence','pose_description','rider_appearance','vehicle_appearance','hand_handlebar_contact','seat_contact','foot_support']}
            combo.update(asset_kind='rider_vehicle_rigid',fixed_pose=True,rider_count=1,pose_usage='uncertain',source_id=uid,shot_id=uid,identity=uid,pose_cluster_id=uid+'_pose',internal_articulation=False,wheel_animation=False,anchor_review=dict(accepted=True,complete_assembly=True,exactly_one_rider=True,seated=True,source=str(folder/'source_review.json')))
            row=dict(scene_id=ROOT.name,obj_id=uid,category=d['category'],image_path=s['anchor'],prompts=[])
            row['prompts']=[prompt(dict(row,assembly=combo))]
            provenance=dict(source_kind='web_photo',source_image=a['source']['path'],source_sha256=a['source']['sha256'],page_url=a['source']['source_page_url'],download_url=a['source']['image_url'],license=a['source']['license'],author=a['source']['author'],metadata=a['source'],source_review=str(folder/'source_review.json'),joint_mask_review=str(folder/'mask_review.json'),camera_calibration=None,identity_scope='one image; cross-photo physical identity unverified',temporal_rigidity='single-image fixed-pose approximation, unverified over time',report_category='riding_vehicle')
            route=dict(scene_id=row['scene_id'],obj_id=uid,category=row['category'],route='H',assembly=combo,source=provenance,real_images=[],real_view_selection=dict(real_count=0,reason='Reviewed joint rider anchor conditions FLUX; H uses reviewed joint-alpha generated image only',duplicate_padding=False))
            assembly(dict(row,assembly=combo))
            write_json(folder/'ready.json',dict(row=row,route=route));write_json(folder/'done.json',dict(status='ready'))
        except Exception as exc:write_json(folder/'done.json',dict(status='preparation_rejected',error=repr(exc)))
    await asyncio.gather(*(finish(a) for a in reviewed))

async def flux_review(client,row,result,folder):
    folder.mkdir(parents=True,exist_ok=True)
    reviews=[];segmented=[]
    for i,img in enumerate(result['images']):
        sub=folder/f'view_{i}';sub.mkdir(exist_ok=True)
        combined=pair(row['image_path'],img['path'],sub/'comparison.jpg')
        decision=await check(client,'Compare A source assembly and B generated assembly. Require the same one seated rider, same vehicle, same fixed pose, plausible hand/seat/foot contacts and complete person plus both wheels. Minor texture details are allowed; reject removed rider, different vehicle type, detached limbs, changed seated pose or duplicated wheels. Boxes are not requested.',combined,PAIR,sub/'review.json')
        if not all(decision[k] for k in ('accepted','complete_assembly','same_pose','same_identity','contacts_preserved')):raise ValueError('FLUX assembly review: '+decision['reason'])
        boxes=await check(client,'Locate the complete person and complete vehicle separately in this image. Return normalized 0..1000 XYXY boxes; vehicle box must contain both wheels and handlebars.',img['path'],LOCATE,sub/'boxes.json')
        await segment([dict(image=img['path'],output=str(sub/'segmentation'),**boxes)],folder.name+f'_flux_{i}')
        s=read_json(sub/'segmentation/result.json')
        if s['status']!='segmented':raise ValueError(s)
        combined=pair(img['path'],s['anchor'],sub/'mask_comparison.jpg')
        m=await check(client,'Compare A complete generated assembly and B joint foreground mask. Require complete rider and vehicle including both tires, head and limbs. Allow tiny spoke/boundary imperfections. Reject missing major components or substantial remaining background.',combined,MASK,sub/'mask_review.json')
        if not all(m[k] for k in MASK['properties'] if k!='reason'):raise ValueError('FLUX mask: '+m['reason'])
        reviews.append(dict(decision,sha256=img['sha256'],source=str(sub/'review.json')))
        segmented.append(dict(path=s['rgba'],sha256=gen.sha256(s['rgba']),flux_sha256=img['sha256'],complete_assembly=True,review=str(sub/'mask_review.json')))
    return dict(flux_assembly_reviews=reviews,flux_segmented_images=segmented)

def support_geometry(entry):
    import numpy as np
    from plyfile import PlyData
    c=entry['canonical'];v=PlyData.read(c['canonical_path'])['vertex']
    xyz=np.stack([v[k] for k in ['x','y','z']],axis=1);opaque=np.asarray(v['opacity'])>0
    size=np.asarray(c['size_xyz']);axis=int(np.argmax(size));metric=float(entry['scale'][['est_length_m','est_width_m','est_height_m'][axis]])/size[axis]
    lo,hi=np.quantile(xyz[:,:2],[.005,.995],axis=0);cx,cy=(lo+hi)/2;length=hi[0]-lo[0]
    zs=[];counts=[]
    for direction in [-1,1]:
        mask=opaque & (direction*(xyz[:,0]-cx)>.2*length) & (abs(xyz[:,1]-cy)<.18*length)
        counts.append(int(mask.sum()));zs.append(float(np.quantile(xyz[mask,2],.005)*metric) if mask.sum()>=32 else None)
    passed=all(z is not None and abs(z)<=.15 for z in zs) and abs(zs[0]-zs[1])<=.15
    return dict(passed=passed,front_rear_lower_quantiles_m=zs,point_counts=counts,metric_scale=metric,dimensions_lwh_m=(size*metric).tolist(),scale_basis='VLM full-assembly estimate, not metric measurement',method='opaque front/rear center strips 0.5-percentile Z; geometric proxy paired with visual tire review, not wheel ground truth',tolerance_m=.15)

async def batch(client,chunk):
    saved=read_json(chunk/'batch.json');rows=saved['rows'];routes=saved['routes'];work=chunk/'generation';work.mkdir(exist_ok=True)
    hf=write_json(chunk/'flux_input.json',rows);rf=write_json(chunk/'routes.json',dict(schema_version=2,assets=routes,flux_input_sha256=gen.sha256(hf)))
    base=gen.prepare_jobs(hf,rf,read_json('/mnt/ojc/DriveLab/configs/generation.example.json'))
    jobs=work/'jobs.json';statepath=work/'generation.json'
    if statepath.exists():
        state=read_json(statepath);bundle=read_json(jobs)
        if state.get('base_signature')!=base['signature']:raise ValueError('Batch input signature changed')
    else:
        bundle=base;state=dict(schema_version=1,signature=base['signature'],base_signature=base['signature'],config=base['config'],assets=[dict(a,stages={},status='pending') for a in base['assets']]);write_json(jobs,bundle);write_json(statepath,state)
    def accept(result):
        r=state['assets'][result['item_index']]
        if result['status']=='succeeded':gen.checked_outputs(result)
        r['stages'][result['stage']]=result;r['status']='generated' if result['stage']=='H' and result['status']=='succeeded' else result['status']
        if r['status']=='generated':r.update({k:result[k] for k in ['raw_path','raw_sha256','raw_up_axis','gaussian_count']})
        write_json(statepath,state)
    async def launch(stage,ids):
        if not ids:return
        gpu=await POOL.acquire(stage)
        try:await gen._launch(stage,ids,gpu,jobs,accept)
        finally:POOL.release(gpu)
    ids=[a['item_index'] for a in state['assets'] if 'flux' not in a['stages']]
    status('flux_generation',batch=chunk.name,items=len(ids))
    await asyncio.gather(*(launch('flux',ids[i::6]) for i in range(min(6,len(ids)))))
    async def review_one(r):
        i=r['item_index']
        if r['stages'].get('flux',{}).get('status')!='succeeded' or r.get('assembly_review_terminal'):return
        try:
            evidence=await flux_review(client,r,r['stages']['flux'],chunk/'assembly_review'/r['obj_id'])
            r.update(evidence);bundle['assets'][i].update(evidence);r['assembly_review_terminal']='accepted'
        except Exception as exc:r.update(status='failed',assembly_review_terminal='rejected',assembly_review_error=repr(exc))
        write_json(statepath,state);write_json(jobs,bundle)
    status('flux_assembly_review',batch=chunk.name)
    await asyncio.gather(*(review_one(r) for r in state['assets']))
    bundle['signature']=gen.digest(dict(base=base['signature'],assets=bundle['assets']));state['signature']=bundle['signature'];write_json(jobs,bundle);write_json(statepath,state)
    ids=[r['item_index'] for r in state['assets'] if r.get('assembly_review_terminal')=='accepted' and 'H' not in r['stages']]
    status('H_reconstruction',batch=chunk.name,items=len(ids))
    await asyncio.gather(*(launch('H',ids[i::6]) for i in range(min(6,len(ids)))))
    state['counts']=dict(queries=len(rows),generated=sum(r['status']=='generated' for r in state['assets']));state['status']='generated_pending_curation';write_json(statepath,state)
    status('render_and_quality',batch=chunk.name,generated=state['counts']['generated'])
    cached=chunk/'curation/curation.json'
    if cached.exists() and (chunk/'library').exists():
        curated=read_json(cached)
    else:
        curated=await curate(statepath,chunk/'library',CONFIG,renderer_python='/home/wmj/miniconda3/envs/sim/bin/python',gpus=GPUS,work_dir=chunk/'curation')
    for uid,r in curated['assets'].items():
        if 'entry' not in r:continue
        e=r['entry'];q=e['quality'];e['provenance']['batch_id']=ROOT.name;e['class']='vehicle'
        if _verdict(q['audit'],q['rendering'],q['heading_decision'],True)!='keep':continue
        try:
            support=await check(client,'Inspect all views of this reconstructed rider+vehicle. Require one complete seated rider, two complete wheels, plausible body-to-seat/hands-to-handlebars/feet-to-pedals-or-footrests contact. Tire bottoms should form a plausible common ground plane; reject floating or severely tilted/unequal wheel support, feet or face as the lowest support instead of tires. Is full assembly length/width/height anatomically plausible? Minor blur/spokes are allowed.',q['rendering']['evidence'],SUPPORT,chunk/'support'/f'{uid}.json')
            geometry=support_geometry(e)
            write_json(chunk/'support'/f'{uid}_geometry.json',geometry)
            q['ground_support_review']=dict(visual=support,geometry=geometry)
            if all(support[k] for k in SUPPORT['properties'] if k!='reason') and geometry['passed']:
                from asset_extraction.rigid_publication import entry_from_metadata
                proposed=dict(e,status='keep');entry_from_metadata(proposed)
                e['status']=r['status']='keep';e.pop('reason',None)
        except Exception as exc:q['ground_support_error']=repr(exc)
    curated['counts']=dict(collections.Counter(r['status'] for r in curated['assets'].values()));write_json(chunk/'curation/final.json',curated)
    kept=dict(curated,assets={k:v for k,v in curated['assets'].items() if v['status']=='keep'})
    published=publish(kept,runtime.LIBRARY,dataset='web',scene_id=ROOT.name) if kept['assets'] else []
    write_json(chunk/'completed.json',dict(counts=curated['counts'],generated=state['counts']['generated'],published=[dict(uid=p['uid'],category=p['category'],status=p['status']) for p in published],kept=sum(p['status']=='keep' and p.get('publication_status')=='published' for p in published),completed=time.time()))

def inventory():
    manifest=read_json('/mnt/ojc/workplace1/asset_work/openimages_riders_20261105/inputs_150_seated/generation_inputs.json')
    if manifest['status']!='source_screened' or manifest['count']<1:
        raise ValueError('Inputs must pass both source screening checks first')
    return [dict(a['source'],approved_input=a['image_path'],approved_assembly_review=a['assembly_review'],approved_seated_review=a['seated_review']) for a in manifest['inputs']]

async def main():
    for d in ['sources','chunks','logs','packets','support']:(ROOT/d).mkdir(exist_ok=True)
    lock=(ROOT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    candidates=inventory();write_json(ROOT/'input_inventory.json',dict(count=len(candidates),category_counts=dict(collections.Counter(a['category'] for a in candidates)),target=100,source='Official OpenImages downloaded photographs; candidate != accepted'))
    client=AsyncOpenAIClient(CONFIG,None)
    try:
        await client.check_endpoints()
        while True:
            candidates=inventory()
            ctl=read_json(ROOT/'control.json')
            completed=[read_json(p) for p in (ROOT/'chunks').glob('*/completed.json')];nkeep=sum(r['kept'] for r in completed)
            write_json(ROOT/'summary.json',dict(updated=time.time(),target_keep=ctl['target_keep'],kept=nkeep,generated=sum(r['generated'] for r in completed),completed_batches=len(completed),counts=dict(sum((collections.Counter(r['counts']) for r in completed),collections.Counter()))))
            if ctl['status']!='running':status('paused_by_control',kept=nkeep);return
            if nkeep>=ctl['target_keep']:status('target_completed',kept=nkeep);return
            if shutil.disk_usage(ROOT).free<ctl['minimum_free_gib']*1024**3:
                status('waiting_disk_space',kept=nkeep);await asyncio.sleep(30);continue
            pending=[p.parent for p in sorted((ROOT/'chunks').glob('*/batch.json')) if not (p.parent/'completed.json').exists() and not (p.parent/'error.json').exists()]
            if pending:
                chunk=pending[0]
                try:await batch(client,chunk)
                except Exception as exc:
                    traceback.print_exc();write_json(chunk/'error.json',dict(error=repr(exc),time=time.time()))
                continue
            used={r['obj_id'] for p in (ROOT/'chunks').glob('*/batch.json') for r in read_json(p)['rows']}
            ready=[read_json(p) for p in sorted((ROOT/'sources').glob('*/ready.json')) if p.parent.name not in used]
            unchecked=[a for a in candidates if not (ROOT/'sources'/('oi_'+a['split']+'_'+a['image_id'])/'done.json').exists()]
            limit=min(6 if not completed else 12,ctl['target_keep']-nkeep)
            if len(ready)>=limit or ready and not unchecked:
                selected=ready[:limit];chunk=ROOT/'chunks'/f'batch_{len(list((ROOT/"chunks").glob("*/batch.json"))):05d}'
                write_json(chunk/'batch.json',dict(rows=[a['row'] for a in selected],routes=[a['route'] for a in selected]));continue
            if unchecked:
                status('source_preparation',kept=nkeep,ready=len(ready),remaining_candidates=len(unchecked))
                await prepare(client,unchecked[:12],f'source_{len(list((ROOT/"sources").glob("*/done.json"))):05d}');continue
            incoming=read_json('/mnt/ojc/workplace1/asset_work/openimages_riders_20261105/inputs_150_seated/generation_inputs.json')
            if incoming['count']<incoming['target']:
                status('waiting_for_more_screened_inputs',kept=nkeep,available_inputs=incoming['count']);await asyncio.sleep(20);continue
            status('source_pool_exhausted',kept=nkeep,shortfall=ctl['target_keep']-nkeep);return
    finally:await client.close()

if __name__=='__main__':
    from direct_policy import install
    install(sys.modules[__name__])
    asyncio.run(main())
