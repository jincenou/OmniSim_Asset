"""Priority bus demos with the existing strict H reconstruction and quality protocol."""
import asyncio,os,sys,time
from pathlib import Path
B=Path(__file__).resolve().parent;P=Path('/mnt/ojc/pre_flux_pipeline')
sys.path.insert(0,str(P));sys.path.insert(0,'/mnt/ojc/workplace1/asset_work/vehicle1000_20261025')
from runtime import process_chunk
from asset_extraction.config import load_config
from asset_extraction.io_utils import read_json,write_json
from asset_extraction.vlm import AsyncOpenAIClient,rigid_vlm_messages,RIGID_RESPONSE_SCHEMA,build_diffusion_prompts
from asset_extraction.real_views import review_references
os.environ['VLM_BASE_URL']='http://127.0.0.1:8082/v1';os.environ['VLM_RESPONSE_FORMAT']='json_object'

async def main():
    config=load_config('/mnt/ojc/workplace1/asset_work/batch20_20261024/frontend_revision_01/configs/waymo_pilot.yaml')
    config.vlm.concurrency=4;config.vlm.max_tokens=2048
    prepared=read_json(B/'prepared.json');client=AsyncOpenAIClient(config,None);sem=asyncio.Semaphore(4)
    async def audit(asset,obs):
        async with sem:
            path=B/'source_audits'/asset['obj_id']/(Path(obs['source_image']).stem+'.json')
            if path.exists():return read_json(path)
            r=await client.call_json(rigid_vlm_messages(obs['image_path'],'bus',obs),response_schema=RIGID_RESPONSE_SCHEMA)
            write_json(path,r);return r
    async def select(asset):
        observations=asset['observations']
        audits=await asyncio.gather(*(audit(asset,o) for o in observations))
        valid=[(o,a) for o,a in zip(observations,audits) if (a['category'],a['asset_form'],a['structure_extent'],a['source_quality'])==('bus','discrete_object','whole_unit','usable') and a['vehicle_view'] in {'front_side','rear_side'}]
        if not valid:return None
        obs,attributes=max(valid,key=lambda pair:pair[0]['quality'])
        row=dict(scene_id=asset['scene_id'],obj_id=asset['obj_id'],category='bus',image_path=obs['image_path'])
        row['prompts']=build_diffusion_prompts(row,attributes)
        source=dict(obs,candidate_id=asset['obj_id'],scene_id=asset['scene_id'],tracklet_id=asset['candidate']['track_id'],
            frame_records=observations,source_tfrecord=asset['candidate']['tfrecord'],context=asset['candidate']['context'],
            native_track_id=asset['candidate']['track_id'],association_method='metadata-assisted native track; DINO boxes and SAM2 masks',annotation_dimensions_used=False)
        route=dict(obj_id=row['obj_id'],scene_id=row['scene_id'],category='bus',route='H',
            real_images=sorted(observations,key=lambda o:-o['quality'])[:4],real_view_selection=None,attributes=attributes,source=source)
        return row,route
    try:
        await client.check_endpoints();selected=await asyncio.gather(*(select(a) for a in prepared))
    finally:await client.close()
    selected=[s for s in selected if s]
    if not selected:raise ValueError('No bus passed source admission')
    rows=[s[0] for s in selected];routes=await review_references([s[1] for s in selected],config,B/'reference_quality.json')
    write_json(B/'status.json',dict(stage='generation',selected=len(rows),real_counts=[len(r['real_images']) for r in routes]))
    result=await process_chunk(B,rows,dict(schema_version=2,assets=routes),config,'waymo','waymo_bus_more_20261025',gpus=[0,2])
    write_json(B/'status.json',dict(stage='completed',**result))

if __name__=='__main__':asyncio.run(main())
