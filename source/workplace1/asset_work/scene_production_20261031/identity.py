"""Dataset-native instance IDs for production dedup; never feed sensor geometry to H."""
import copy,importlib.util,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
from runtime import ROOT,P,LIBRARY,read_json,write_json
from asset_extraction.vision import _bbox_iou
from asset_extraction.real_views import VEHICLES,diverse_order
OLD=Path('/mnt/ojc/workplace1/asset_work/batch20_20261024')

def camera_links(scene):
    """Link camera IDs only when >=2 native projected-box matches agree without contradiction."""
    path=ROOT/'identity'/f'{scene}_camera_links.json'
    if path.exists():return read_json(path)['links']
    import struct
    from collections import Counter
    sys.path.insert(0,'/mnt/ojc/data/waymo_track_bbox_10/python_pkgs')
    from waymo_open_dataset import dataset_pb2
    raw=next(r['raw_tfrecord'] for r in read_json(ROOT/'registry.json')['waymo'] if r['scene_id']==scene)
    names={1:'front',2:'front_left',3:'front_right',4:'side_left',5:'side_right'}
    observations=defaultdict(list);frame_index=0
    def box(b):return [b.center_x-b.length/2,b.center_y-b.width/2,b.center_x+b.length/2,b.center_y+b.width/2]
    with open(raw,'rb') as stream:
        while header:=stream.read(12):
            payload=stream.read(struct.unpack('<Q',header[:8])[0]);assert len(stream.read(4))==4
            if frame_index%5==0:
                frame=dataset_pb2.Frame();frame.ParseFromString(payload)
                vehicles={r.id for r in frame.laser_labels if r.type==1};projected={}
                for group in frame.projected_lidar_labels:
                    projected[group.name]=[(box(r.box),oid) for r in group.labels for oid in vehicles if r.id.startswith(oid+'_')]
                for camera in frame.camera_labels:
                    for row in camera.labels:
                        if row.type!=1:continue
                        ranks=sorted([(_bbox_iou(box(row.box),bbox),oid) for bbox,oid in projected.get(camera.name,[])],reverse=True)
                        if ranks and ranks[0][0]>=.55 and (len(ranks)==1 or ranks[0][0]-ranks[1][0]>=.15):
                            observations[names[camera.name]+':'+row.id].append(dict(frame=frame_index,iou=ranks[0][0],id=ranks[0][1]))
            frame_index+=1
    links={key:dict(id=values[0]['id'],evidence=values,support=len(values)) for key,values in observations.items()
           if len(values)>=2 and len({r['id'] for r in values})==1}
    write_json(path,dict(protocol='native camera-to-projected-LiDAR identity link; >=2 consistent observations, no conflicting match',source_tfrecord=raw,links=links))
    return links

def canonical_identity(info,dataset,scene):
    result=copy.deepcopy(info);native=result.get('native_id')
    if dataset=='waymo' and native and ':laser_association:' in native:
        result['native_id']=native.replace(':laser_association:',':lidar_track:',1)
        native=result['native_id']
    if dataset=='waymo' and native and ':camera:' in native:
        path=ROOT/'identity'/f'{scene}_camera_links.json'
        links=read_json(path)['links'] if path.exists() else {}
        key=native.split(':camera:',1)[1]
        if key in links:
            result['camera_scoped_id']=native;result['native_id']=f'waymo:{scene}:lidar_track:{links[key]["id"]}'
            result['cross_camera_link_support']=links[key]['support']
    return result

def records(dataset,scene):
    existing=OLD/'evaluation/ground_truth'/f'{scene}.json'
    if existing.exists():
        return [r for r in read_json(existing)['records'] if r['category'] in VEHICLES]
    if dataset=='waymo':
        result=[]
        path=ROOT/'evaluation/waymo_native'/f'{scene}.jsonl'
        with path.open() as f:
            for line in f:
                for camera in json.loads(line).get('camera_annotations',[]):
                    for row in camera['annotations']:
                        if row['type']!=1:continue
                        linked=row.get('laser_object_id')
                        identity=f'laser_association:{linked}' if linked else f'camera:{camera["camera_name"]}:{row["camera_track_id"]}'
                        result.append(dict(id=f'waymo:{scene}:{identity}',source_image=camera['path'],bbox_xyxy=row['bbox_xyxy']))
        return result
    import ijson
    spec=importlib.util.spec_from_file_location('native_projection',OLD/'scripts/prepare_evaluation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    root=Path('/mnt/public_data/nuScenes/v1.0-trainval');frames=read_json(ROOT/'inputs/nuscenes/debug/dataset_scenes'/f'{scene}.json')['frames']
    categories={r['token']:r['name'] for r in read_json(root/'category.json')}
    instances={r['token']:categories[r['category_token']] for r in read_json(root/'instance.json')}
    wanted={f['sample_token'] for f in frames};annotations=defaultdict(list)
    with (root/'sample_annotation.json').open('rb') as handle:
        for r in ijson.items(handle,'item',use_float=True):
            if r['sample_token'] in wanted and module.nusc_category(instances[r['instance_token']]) in VEHICLES:annotations[r['sample_token']].append(r)
    result=[]
    for frame in frames:
        with Image.open(frame['path']) as im:w,h=im.size
        for row in annotations[frame['sample_token']]:
            transform=np.eye(4);transform[:3,:3]=module.rotation(row['rotation']);transform[:3,3]=row['translation']
            bbox=module.project_box(transform,[row['size'][1],row['size'][0],row['size'][2]],frame,w,h)
            if bbox:result.append(dict(id=f'nuscenes:{scene}:{row["instance_token"]}',source_image=frame['path'],bbox_xyxy=bbox))
    return result

def match_observation(obs,index):
    ranked=sorted([(_bbox_iou(obs['bbox_xyxy'],r['bbox_xyxy']),r['id']) for r in index.get(Path(obs['source_image']).name,[])],reverse=True)
    if not ranked or ranked[0][0]<.55 or (len(ranked)>1 and ranked[0][0]-ranked[1][0]<.15):return None
    return dict(native_id=ranked[0][1],iou=ranked[0][0],source_image=obs['source_image'])

def prepare_identity(dataset,scene,run):
    path=ROOT/'identity'/f'{dataset}_{scene}.json'
    if dataset=='waymo':camera_links(scene)
    if path.exists():
        result=read_json(path)
        result['matches']={key:canonical_identity(value,dataset,scene) for key,value in result['matches'].items()}
        result['existing_assets']={key:canonical_identity(value,dataset,scene) for key,value in result.get('existing_assets',{}).items()}
        for info in result['matches'].values():
            info['reference_matches']={key:canonical_identity(value,dataset,scene) if value else None for key,value in info['reference_matches'].items()}
        result['identity_version']=2;write_json(path,result);return result
    index=defaultdict(list)
    for row in records(dataset,scene):index[Path(row['source_image']).name].append(row)
    routes=read_json(run/'handoff/asset_routes.json')['assets'];matches={}
    for route in routes:
        if route['category'] not in VEHICLES:continue
        source=route['source'];observations={}
        for obs in [source,*source.get('frame_records',[])]:
            observations[(obs['source_image'],tuple(obs['bbox_xyxy']))]=obs
        evidence=[m for obs in observations.values() if (m:=match_observation(obs,index))]
        ids={r['native_id'] for r in evidence}
        matches[route['obj_id']]=dict(native_id=next(iter(ids)) if len(ids)==1 else None,
            status='matched' if len(ids)==1 else ('conflicting_native_identities' if ids else 'unmatched'),
            evidence=evidence,reference_matches={r['sha256']:match_observation(r,index) for r in route['real_images']})
    prior={};raw_scene=None
    if dataset=='waymo':raw_scene=next(r['raw_tfrecord'] for r in read_json(ROOT/'registry.json')['waymo'] if r['scene_id']==scene)
    for row in read_json(LIBRARY/'manifests/asset_library_rigid.json')['assets']:
        if row['dataset']!=dataset or row['category'] not in VEHICLES:continue
        source=row['provenance']['source']
        if row['provenance']['scene_id']!=scene and not (raw_scene and source.get('source_tfrecord')==raw_scene):continue
        evidence=[m for obs in [source,*source.get('frame_records',[])] if (m:=match_observation(obs,index))]
        ids={m['native_id'] for m in evidence}
        if len(ids)==1:prior[row['uid']]=dict(native_id=next(iter(ids)),status=row['status'])
    matches={key:canonical_identity(value,dataset,scene) for key,value in matches.items()}
    for info in matches.values():
        info['reference_matches']={key:canonical_identity(value,dataset,scene) if value else None for key,value in info['reference_matches'].items()}
    payload=dict(identity_version=2,dataset=dataset,scene_id=scene,protocol='metadata-assisted production identity deduplication; not an RGB-only benchmark',
        sensor_geometry_used_for_reconstruction=False,iou_threshold=.55,runner_up_margin=.15,matches=matches,
        existing_assets={key:canonical_identity(value,dataset,scene) for key,value in prior.items()})
    write_json(path,payload);return payload

def distinct_selection(rows,routes,mapping,reserved=(),occupied_native=()):
    byid={r['obj_id']:r for r in routes};selected=[];groups=defaultdict(list);decisions=[]
    occupied={mapping[oid]['native_id'] for oid in reserved if oid in mapping and mapping[oid]['native_id']}
    occupied.update(occupied_native)
    for row in rows:
        info=mapping[row['obj_id']];native=info['native_id']
        if info['status']=='conflicting_native_identities':
            decisions.append(dict(obj_id=row['obj_id'],reason=info['status']));continue
        if native in occupied:
            decisions.append(dict(obj_id=row['obj_id'],reason='native_identity_already_scheduled',native_id=native));continue
        groups[native or row['obj_id']].append(row)
    output_routes=[]
    for identity,group in groups.items():
        def rank(row):
            route=byid[row['obj_id']];pixels=route['source'].get('native_foreground',{}).get('pixels',0)
            return (bool(route['real_images']),pixels,len(route['real_images']))
        chosen=max(group,key=rank);route=copy.deepcopy(byid[chosen['obj_id']]);info=mapping[chosen['obj_id']]
        route['source']['production_identity']=dict(native_id=info['native_id'],status=info['status'],protocol='metadata-assisted identity only')
        pooled={}
        for row in group:
            oid=row['obj_id']
            for ref in byid[oid]['real_images']:
                linked=mapping[oid]['reference_matches'].get(ref['sha256'])
                if linked and linked['native_id']==info['native_id'] and ref.get('quality_review',{}).get('accepted'):pooled[ref['sha256']]=dict(ref,rgba_sha256=ref['sha256'])
            if oid!=chosen['obj_id']:decisions.append(dict(obj_id=oid,duplicate_of=chosen['obj_id'],native_id=info['native_id']))
        if info['native_id']:
            refs=list(pooled.values())
            route['real_images']=diverse_order(refs)[:4]
            count=len(route['real_images'])
            route['real_view_selection']=dict(real_count=count,preferred_count=[3,4],maximum=4,
                status='preferred' if count>=3 else ('two_view' if count==2 else ('single_view_limited' if count else 'flux_only')),
                reason='quality-approved references pooled only across the same verified native identity',duplicate_padding=False)
        selected.append(chosen);output_routes.append(route)
    return selected,output_routes,decisions

if __name__=='__main__':
    dataset,scene,run=sys.argv[1:];r=prepare_identity(dataset,scene,Path(run))
    print(json.dumps(dict(objects=len(r['matches']),matched=sum(m['status']=='matched' for m in r['matches'].values()))))
