"""Two distinct Waymo buses through the unchanged source audit, FLUX, H and curation."""
import asyncio, os, sys, json, time, importlib.util
from copy import deepcopy
from pathlib import Path
P=Path('/mnt/ojc/pre_flux_pipeline');B=Path(__file__).resolve().parent
OLD=Path('/mnt/ojc/workplace1/asset_work/batch20_20261024/runs/waymo10')
N=Path('/mnt/ojc/workplace1/asset_work/batch20_20261024/frontend_revision_01')
sys.path.insert(0,str(P))
from asset_extraction.config import load_config
from asset_extraction.io_utils import read_json,write_json,write_manifest
from asset_extraction.vlm import AsyncOpenAIClient,rigid_vlm_messages,RIGID_RESPONSE_SCHEMA,build_diffusion_prompts
from asset_extraction.generation import prepare_jobs,generate,sha256
from asset_extraction.export import curate
os.environ['VLM_BASE_URL']='http://127.0.0.1:8082/v1'
os.environ['VLM_RESPONSE_FORMAT']='json_object'
os.environ['ASSET_BATCH_ROOT']=str(B)
IDS=['waymo_val_003_t000055','waymo_val_005_t000015']
async def main():
    config=load_config(N/'configs/waymo_pilot.yaml'); config.vlm.concurrency=2
    config.vlm.max_tokens=2048
    source_manifest=read_json(OLD/'merged/manifest.filtered.json');routing=read_json(OLD/'handoff/asset_routes.json')
    selected=[deepcopy(next(r for r in source_manifest if r['obj_id']==oid)) for oid in IDS]
    routes=[deepcopy(next(r for r in routing['assets'] if r['obj_id']==oid)) for oid in IDS]
    client=AsyncOpenAIClient(config,None)
    try:
        await client.check_endpoints()
        async def check(row,route):
            p=B/'source_audit'/f'{row["obj_id"]}.json'
            if p.exists():return read_json(p)
            a=await client.call_json(rigid_vlm_messages(row['image_path'],row['category'],route['source']),response_schema=RIGID_RESPONSE_SCHEMA)
            write_json(p,a);return a
        audits=await asyncio.gather(*(check(row,route) for row,route in zip(selected,routes)))
    finally:await client.close()
    for row,route,a in zip(selected,routes,audits):
        if (a['category'],a['asset_form'],a['structure_extent'],a['source_quality'])!=('bus','discrete_object','whole_unit','usable'):
            raise ValueError(f'Bus source did not pass original admission: {row["obj_id"]}: {a}')
        row['prompts']=build_diffusion_prompts(row,a)
        route['attributes']=a
    handoff=write_manifest(B/'inputs/flux_input.json',selected)
    sidecar=write_json(B/'inputs/asset_routes.json',dict(routing,flux_input=str(handoff),flux_input_sha256=sha256(handoff),assets=routes))
    write_json(B/'selection.json',dict(objects=IDS,distinct_objects=True,selection='visual confirmation across two different Waymo scenes',real_counts=[len(r['real_images']) for r in routes],reused='GroundingDINO/SAM/association outputs with source and mask provenance; real references already quality approved',fresh='current source attribute audit and prompts, FLUX images, H reconstruction, canonicalization, sixteen-view rendering, original strict quality gate'))
    jobs=prepare_jobs(handoff,sidecar,read_json(P/'configs/generation.example.json'))
    write_json(B/'status.json',dict(stage='generation',started=time.time()))
    await generate(jobs,B/'generation',[0,1,7])
    write_json(B/'status.json',dict(stage='curation'))
    result=await curate(B/'generation/generation.json',B/'library',config,renderer_python='/home/wmj/miniconda3/envs/sim/bin/python',gpus=[0,1,7],work_dir=B/'curation')
    write_json(B/'status.json',dict(stage='publishing',counts=result.get('counts'),curation_status=result['status']))
    spec=importlib.util.spec_from_file_location('bus_batch_publish','/mnt/ojc/workplace1/asset_work/batch20_20261024/scripts/run_batch.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.BATCH_ID='waymo_bus_20261025'
    published=mod.publish(B/'library','waymo','bus_demo')
    write_json(B/'published.json',published)
    write_json(B/'status.json',dict(stage='completed',counts=result['counts'],curation_status=result['status'],published=published,finished=time.time()))
    print(json.dumps(dict(counts=result['counts'],published=published),ensure_ascii=False),flush=True)
asyncio.run(main())
