"""Web-only final metrics using the existing fixed foreground evaluation protocol."""
import asyncio
import copy
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRODUCTION = ROOT.parent / 'scene_production_20261031'
sys.path.insert(0, str(PRODUCTION))
import runtime
spec = importlib.util.spec_from_file_location('shared_production_metrics', PRODUCTION / 'metrics.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
shared.ROOT = ROOT
shared.SCRIPT = ROOT / 'evaluate.py'
shared.SNAPSHOT = 'web_final_vehicles'
shared.REGISTRY = ROOT / 'evaluation/metric_registry_web_final_vehicles.json'
shared.GROUP = 'vehicles'
os.environ['VEHICLE_METRIC_GPUS'] = '0,1,2,3,5,6'

def freeze():
    if shared.REGISTRY.exists():
        return runtime.read_json(shared.REGISTRY)
    state = runtime.read_json(ROOT / 'status.json')
    if state['status'] != 'first_wave_finished':
        raise RuntimeError('Web wave must finish before freezing final metrics')
    batches = sorted(ROOT.glob('chunks/*/batch.json'))
    rows = []
    queries = sum(len(runtime.read_json(p)['rows']) for p in batches)
    for batch in batches:
        chunk = batch.parent
        if not ((chunk / 'completed.json').exists() or (chunk / 'error.json').exists()):
            raise RuntimeError('Unfinished web batch: ' + str(chunk))
        curation = chunk / 'curation/curation.json'
        if not curation.exists():
            continue
        for oid, result in runtime.read_json(curation)['assets'].items():
            if 'entry' not in result:
                continue
            entry = result['entry']
            source = ROOT / 'sources' / oid
            mask = runtime.read_json(source / 'segmentation/result.json')
            canonical = copy.deepcopy(entry['canonical'])
            rows.append(dict(obj_id=oid, dataset='web', scene_id='commons',
                route=entry['route'], category=entry['category'], quality_status=result['status'],
                canonical=canonical, ply=canonical['canonical_path'], ply_sha256=canonical['sha256'],
                source_rgba=mask['rgba_path'], source_sha256=runtime.sha256(mask['rgba_path']),
                source_crop_box=mask['crop_box'], native_dimensions_annotation=None,
                matched_annotation=None, matched_exclusion='Web photograph has no calibrated camera or native cuboid pose'))
    protocol = copy.deepcopy(runtime.read_json(shared.OLD / 'frontend_revision_01/evaluation/metric_registry.json')['protocol'])
    for ckpt in protocol['checkpoints'].values():
        if runtime.sha256(ckpt['path']) != ckpt['sha256']:
            raise ValueError('Evaluation checkpoint hash changed')
    protocol.update(version='web_vehicles_source_foreground_v1', queries=queries,
        exported_payloads=len(rows), created=time.time(), partial_snapshot=False,
        script_sha256=runtime.sha256(shared.SCRIPT), snapshot_script_sha256=runtime.sha256(__file__),
        source='One original SAM foreground per Commons source file; source-conditioned, not held-out.',
        identity='Exact/perceptual image duplicates suppressed; different photographs can show the same physical vehicle. No cross-photo native identity is claimed.',
        cohorts='All valid recorded exports and final keep separately, including review/reject records. Failed queries remain in web AOR.',
        coverage='Unavailable for web photographs; web sources are excluded from nuScenes/Waymo coverage.',
        comparisons='Web sources form a separate population and are never pooled into nuScenes/Waymo scores.',
        matched_camera='Unavailable; matched ED-R, PSNR, SSIM, LPIPS, silhouette and native size metrics are N/A.',
        FAD='Not used. User approved separately named FD-DINO.',
        FD_DINO='Frechet distance of L2-normalized final DINOv3 CLS, one source versus 16 orbit renders per identity. Unbiased covariance, float64 statistics, minimum two source identities. Not FAD.',
        withdrawn_pilot='Source-correction pilot retained under validation, excluded from final production cohort; no outputs from that pilot were published.')
    result = dict(protocol=protocol, assets=rows)
    runtime.write_json(shared.REGISTRY, result)
    return result

shared.freeze = freeze
if __name__ == '__main__':
    asyncio.run(shared.main())
