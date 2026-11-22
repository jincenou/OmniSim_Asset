"""One fixed new-scene input exercises real production backends, without changing its live queue."""
import asyncio
import copy
import json
import logging
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import runtime  # Install the same shared GPU lease allocator as production.
from asset_extraction.config import load_config
from asset_extraction.pipeline import _apply_scene_vlm
from asset_extraction.generation import prepare_jobs,generate,sha256
from asset_extraction.export import curate
from asset_extraction.bank import publish
from asset_extraction.io_utils import read_json,write_json
from nuscenes.quality_filter import filter_manifest
from nuscenes.prepare_handoff import prepare
from nuscenes.prepare_routes import prepare_routes

SCENE='waymo_val_030'
OID='waymo_val_030_t001111'
WORK=ROOT/'validation/end_to_end_smoke'
os.environ['VLM_BASE_URL']='http://127.0.0.1:8082/v1'
os.environ['VLM_RESPONSE_FORMAT']='json_object'
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'


async def main():
    if (WORK/'completed.json').exists():
        raise RuntimeError('Smoke already completed; inspect retained evidence instead of repeating it')
    config=load_config(ROOT/f'configs/waymo_{SCENE}.yaml')
    scene_dir=WORK/'scenes'/SCENE
    source_dir=ROOT/f'runs/waymo_{SCENE}/scenes/{SCENE}'
    source=next(x for x in read_json(source_dir/'logs/export_items_full.json') if x['asset_id']==OID)
    # Build from the actual immutable source record, independent of a concurrent VLM manifest rewrite.
    row=dict(scene_id=SCENE,obj_id=OID,category=source['detected_category'],image_path=source['image_path'],prompts=[])
    plan=WORK/'plan.json'
    if not plan.exists():
        write_json(plan,dict(obj_id=OID,source_image_sha256=sha256(row['image_path']),
            selection='Highest mask quality, then pre-score among detected cars in first Waymo scene; fixed before inference.',
            purpose='Deployment validation only; excluded from production AOR, coverage and target counts.',
            source_record=source,created=time.time(),central_catalogue_mutation=False))
    else:
        assert read_json(plan)['source_image_sha256']==sha256(row['image_path'])
    manifest=scene_dir/'diffusion_inputs/manifest.json'
    write_json(scene_dir/'logs/export_items_full.json',[copy.deepcopy(source)])
    write_json(manifest,[row])
    status=await _apply_scene_vlm(config,manifest,logging.getLogger('smoke'))
    write_json(WORK/'source_audit.json',status)
    filtered=WORK/'merged/manifest.filtered.json'
    filter_manifest(manifest,filtered,WORK/'logs/quality_rejects.jsonl',run_dir=WORK)
    if not read_json(filtered):
        write_json(WORK/'stopped.json',dict(reason='Fixed smoke input rejected by unchanged source checks',time=time.time()))
        return
    handoff=WORK/'handoff/flux_input.json';routes=WORK/'handoff/asset_routes.json'
    prepare(filtered,handoff)
    await prepare_routes(handoff,WORK,routes,config)
    bundle=prepare_jobs(handoff,routes,read_json(runtime.P/'configs/generation.example.json'))
    await generate(bundle,WORK/'generation',runtime.GPUS)
    state=await curate(WORK/'generation/generation.json',WORK/'canonical_library',config,
        renderer_python='/home/wmj/miniconda3/envs/sim/bin/python',gpus=runtime.GPUS,work_dir=WORK/'curation')
    delivery=WORK/'delivery'
    published=publish(state,delivery,dataset='waymo',scene_id=SCENE)
    second=publish(state,delivery,dataset='waymo',scene_id=SCENE)
    assert len(published)==len(second)
    for row in published:
        package=delivery/row['package_path'];receipt=read_json(package/'package.json')
        assert all(sha256(package/path)==digest for path,digest in receipt['files'].items())
    runtime.preserve_outputs(WORK,state)
    generated=read_json(WORK/'generation/generation.json')
    valid=all(x.get('entry',{}).get('quality',{}).get('rendering',{}).get('all_16_nonempty',False) for x in state['assets'].values())
    write_json(WORK/'completed.json',dict(completed=time.time(),obj_id=OID,source=status,
        generation=generated['counts'],curation=state['counts'],published=len(published),all_16_nonempty=valid,
        package_roundtrip_checked=bool(published),production_counted=False,central_catalogue_mutated=False))
    print(json.dumps(read_json(WORK/'completed.json')),flush=True)


if __name__=='__main__':
    asyncio.run(main())
