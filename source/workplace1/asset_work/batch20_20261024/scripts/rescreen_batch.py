"""Reuse verified pilot extraction, then screen the complete fixed 20-scene cohort."""
from __future__ import annotations
import asyncio, json, os, shutil, sys, time
from pathlib import Path
import yaml

P=Path('/mnt/ojc/pre_flux_pipeline')
B=Path(__file__).resolve().parents[1]
N=B/'frontend_revision_01'
sys.path.insert(0,str(P))
from asset_extraction.config import load_config
from asset_extraction.pipeline import ImagePipeline
from asset_extraction.io_utils import read_json,write_json
from asset_extraction.generation import sha256


def prepare(dataset):
    pilot=N/f'runs/{dataset}_pilot'
    handoff=pilot/'handoff/flux_input.json';routes=read_json(pilot/'handoff/asset_routes.json')
    if routes['flux_input_sha256'] != sha256(handoff):
        raise ValueError('Pilot route/manifest mismatch')
    registry=read_json(B/'registry.json')
    scenes=registry['nuscenes'] if dataset=='nuscenes' else [r['scene_id'] for r in registry['waymo']]
    cfg=yaml.safe_load((N/f'configs/{dataset}_pilot.yaml').read_text())
    cfg.update(run_name=f'{dataset}10',scene_ids=scenes,gpu_ids=[3,5,6] if dataset=='nuscenes' else [4,7])
    path=N/f'configs/{dataset}.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    pipeline=ImagePipeline(load_config(path),path);pipeline._claim_run()
    full=N/f'runs/{dataset}10'
    old=read_json(pilot/'input_signature.json');new=read_json(full/'input_signature.json')
    if old['code']!=new['code'] or any(old['config'][key]!=value for key,value in new['config'].items()
                                     if key not in {'scene_ids','limit_scenes'}):
        raise ValueError('Pilot cannot be reused under changed source-processing code/settings')
    scene=scenes[0];source=pilot/'scenes'/scene;target=full/'scenes'/scene
    status=read_json(source/'status.json')
    if status.get('status')!='completed' or not status.get('vlm'):
        raise ValueError('Pilot extraction/attribute pass is not complete')
    if not target.exists():
        shutil.copytree(source,target)
    write_json(full/'reports/pilot_reuse.json',dict(scene=scene,source=str(source),
        pilot_signature=sha256(pilot/'input_signature.json'),target=str(target),
        reason='same extraction code and settings, expanded only the fixed scene list',
        source_paths_remain_in_pilot=True))
    return path,full,cfg['gpu_ids']


async def run(dataset):
    path,full,gpus=prepare(dataset)
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',VLM_RESPONSE_FORMAT='json_schema',
        CONFIG=str(path),OUTPUT_ROOT=str(N),RUN_NAME=f'{dataset}10',GPUS=','.join(map(str,gpus)))
    log=B/f'logs/frontend_{dataset}_full.log'
    with log.open('a') as handle:
        process=await asyncio.create_subprocess_exec('bash','scripts/run_nuscenes.sh','--handoff-only',
            cwd=P,env=env,stdout=handle,stderr=asyncio.subprocess.STDOUT)
        write_json(full/'reports/coordinator.json',dict(pid=process.pid,log=str(log),started=time.time()))
        code=await process.wait()
    if code:
        raise RuntimeError(f'{dataset} frontend failed ({code}); inspect {log}')
    rows=read_json(full/'handoff/flux_input.json')
    write_json(N/f'reports/{dataset}_frontend_complete.json',dict(queries=len(rows),scenes=10,
        handoff=str(full/'handoff/flux_input.json'),sha256=sha256(full/'handoff/flux_input.json')))
    print(dataset,'frontend complete',len(rows),flush=True)


async def main():
    await asyncio.gather(run('nuscenes'),run('waymo'))


if __name__=='__main__':
    asyncio.run(main())
