"""Audit source admission and RGB association without generating new assets."""
from __future__ import annotations
import argparse, asyncio, copy, json, logging, sys, time
from collections import Counter
from pathlib import Path

P = Path('/mnt/ojc/pre_flux_pipeline')
B = Path(__file__).resolve().parents[1]
OUT = B/'frontend_revision_01/reports'
sys.path.insert(0,str(P))
from asset_extraction.io_utils import write_json


async def source_audit():
    from asset_extraction.config import load_config
    from asset_extraction.vlm import AsyncOpenAIClient, rigid_vlm_messages, RIGID_RESPONSE_SCHEMA
    rows=json.loads((B/'runs/nuscenes10/scenes/scene-0001/logs/export_items_full.json').read_text())
    ids={1,2,3,5,7,8,10,20,31,40,60,70,81,82,83,84,100,130,150,195,199,200,201,202,203,204,210,220,230,231,232,233,260,269,270,271,272,273,274,275,276}
    selected=[r for r in rows if int(r['asset_id'].split('t')[-1]) in ids]
    client=AsyncOpenAIClient(load_config(B/'configs/nuscenes.yaml'),logging.getLogger('audit'))
    started=time.monotonic()
    try:
        results=await client.call_many_json([rigid_vlm_messages(r['image_path'],r['detected_category'],r['source'])
            for r in selected],concurrency=4,response_schema=RIGID_RESPONSE_SCHEMA,collect_errors=True)
    finally:
        await client.close()
    records=[dict(obj_id=r['asset_id'],image_path=r['image_path'],source=r['source'],old_category=r.get('vlm',{}).get('category'),
        result=v if isinstance(v,dict) else dict(error=f'{type(v).__name__}: {v}')) for r,v in zip(selected,results)]
    write_json(OUT/'source_admission_pilot.json',dict(elapsed_sec=time.monotonic()-started,
        note='Diagnostic sample chosen to include observed defects; not an unbiased rejection-rate estimate.',
        counts=dict(Counter(('accepted' if r['result'].get('asset_form')=='discrete_object' and r['result'].get('structure_extent')=='whole_unit' and r['result'].get('source_quality')=='usable' else r['result'].get('asset_form','error')+'/'+r['result'].get('structure_extent','error')) for r in records)),assets=records))
    print('source',dict(Counter(('accepted' if r['result'].get('asset_form')=='discrete_object' and r['result'].get('structure_extent')=='whole_unit' and r['result'].get('source_quality')=='usable' else r['result'].get('asset_form','error')+'/'+r['result'].get('structure_extent','error')) for r in records)),flush=True)


async def dedicated_audit():
    from asset_extraction.config import load_config
    from asset_extraction.vlm import AsyncOpenAIClient,rigid_vlm_messages
    rows=json.loads((OUT/'source_admission_pilot.json').read_text())['assets']
    fields=dict(unit_type=['independent_object','functional_head_or_sign_plate','continuous_structure',
        'attached_fragment','standing_vegetation','other'],
        mask_extent=['complete_visible_unit','missing_major_structure','uncertain'],
        lower_termination=['visible_physical_base','hidden_by_occluder','continues_outside_mask','not_applicable','uncertain'],
        detail=['usable','insufficient'])
    schema=dict(type='object',additionalProperties=False,properties={
        **{k:dict(type='string',enum=v) for k,v in fields.items()},'reason':dict(type='string')},
        required=[*fields,'reason'])
    prompt=('Inspect the original scene and the exact segmented target for admission BEFORE 3D reconstruction. '
        'This is a quality audit, not image description or generation. Red boxes and lines are annotations, not physical occluders. '
        'LEFT is the SAM mask target, RIGHT its original scene. If bottom panels exist, they magnify its upper/lower endpoints. '
        'Classify the actual physical unit: a curb/fence/rail strip is continuous_structure; upright trees are standing_vegetation; '
        'an isolated shaft slice, arm or base attached to a larger structure is attached_fragment. '
        'A complete traffic signal HEAD (even its side/back) and a complete sign PLATE are functional_head_or_sign_plate '
        'without their supporting pole; do not reject these for absent poles. '
        'For a whole pole/streetlamp, the mask must include the entire shaft down to its physical ground/base termination '
        'and its top termination/lamp. Trace the lower shaft in the raw scene. If it goes behind a car/foliage, choose '
        'hidden_by_occluder; if the physical shaft visibly continues below the mask, choose continues_outside_mask. '
        'A convenient segmentation endpoint is not evidence of a physical base. These cases have missing_major_structure. '
        'For signs/heads/cars/bins/hydrants/cones and other objects set lower_termination=not_applicable. '
        'Judge completeness only for the observed viewpoint, never demand the unseen back. Minor local occlusion is allowed; '
        'most of the body or essential outline missing is not. detail=insufficient when original pixels do not resolve structure. '
        'Report the evidence briefly. Do not predict whether a generator can imagine the missing parts.')
    batches=[]
    for row in rows:
        msg=rigid_vlm_messages(row['image_path'],row['old_category'],row['source'])
        msg[0]['content'][0]['text']=prompt
        batches.append(msg)
    config=load_config(B/'configs/nuscenes.yaml');client=AsyncOpenAIClient(config,logging.getLogger('audit'))
    started=time.monotonic()
    try:results=await client.call_many_json(batches,concurrency=4,response_schema=schema,collect_errors=True)
    finally:await client.close()
    def accepted(v):
        return isinstance(v,dict) and v['unit_type'] in {'independent_object','functional_head_or_sign_plate'} and (
            v['mask_extent']=='complete_visible_unit' and v['lower_termination'] in {'visible_physical_base','not_applicable'}
            and v['detail']=='usable')
    record=dict(seconds=time.monotonic()-started,schema=schema,prompt=prompt,
        assets=[dict(obj_id=row['obj_id'],accepted=accepted(v),result=v if isinstance(v,dict) else dict(error=str(v)))
                for row,v in zip(rows,results)])
    write_json(OUT/'dedicated_admission_pilot.json',record)
    print('dedicated',record['seconds'],dict(Counter(r['accepted'] for r in record['assets'])),flush=True)


def tracking_audit():
    import cv2
    cv2.setNumThreads(2)
    from asset_extraction.vision import _tracklets, _cross_camera_tracklets, ImageCache, _bbox_iou
    start=time.monotonic()
    regression=json.loads((P/'outputs/validation/vehicle_identity_regression.json').read_text())
    result={group:{name:[[r['frame_id'] for r in t] for t in _tracklets(copy.deepcopy(rows),image_cache=ImageCache(8))]
        for name,rows in regression[group].items()} for group in ('negative_groups','positive_groups')}
    write_json(OUT/'vehicle_identity_regression.json',result)
    if any(any(len(t)>1 for t in partitions) for partitions in result['negative_groups'].values()):
        raise ValueError('A previously reviewed mixed-vehicle negative group was joined')
    scene=B/'runs/nuscenes10/scenes/scene-0001'
    rows=json.loads((scene/'logs/detections.json').read_text())
    tracks=_tracklets(rows,image_cache=ImageCache(8))
    tracks,cross=_cross_camera_tracklets(tracks)
    items=json.loads((scene/'logs/export_items_full.json').read_text())
    indexed={}
    for index,track in enumerate(tracks):
        for row in track:
            indexed.setdefault((row['camera_name'],row['frame_id'],row['category']),[]).append((index,row))
    mapping=[]
    for item in items:
        source=item['source']
        matches=[(index,_bbox_iou(source['bbox_xyxy'],row['bbox_xyxy'])) for index,row in indexed.get(
            (source['camera_name'],source['frame_index'],item['detected_category']),[])]
        best=max(matches,key=lambda v:v[1]) if matches else (None,0)
        mapping.append(dict(obj_id=item['asset_id'],new_track=best[0] if best[1]>.99 else None,
                            category=item.get('vlm',{}).get('category'),iou=best[1]))
    result=dict(scene='scene-0001',old=json.loads((scene/'status.json').read_text())['num_tracklets'],
        new=len(tracks),cross_camera_merges=cross,elapsed_sec=time.monotonic()-start,
        counts=dict(Counter(t[0]['category'] for t in tracks)),source_mapping=mapping,
        track_lengths=dict(Counter(str(min(4,len(t))) for t in tracks)))
    write_json(OUT/'tracking_scene0001.json',result)
    write_json(OUT/'tracking_scene0001_tracks.json',tracks)
    print('tracking',{k:v for k,v in result.items() if k!='source_mapping'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['source','tracking','dedicated'])
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.mode=='source':asyncio.run(source_audit())
    elif args.mode=='dedicated':asyncio.run(dedicated_audit())
    else:tracking_audit()
