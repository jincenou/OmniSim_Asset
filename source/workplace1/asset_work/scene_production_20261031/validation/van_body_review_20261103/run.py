"""Isolate one visually verified body-completeness miss, without stopping production."""
import asyncio
from collections import Counter
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path('/mnt/ojc/workplace1/asset_work/scene_production_20261031')
WORK = Path(__file__).resolve().parent
CHUNK = ROOT / 'chunks/waymo_waymo_val_033_0120'
UID = 'waymo_val_033_t000754'
GLOBAL_UID = 'waymo:waymo_val_033:' + UID
EXPECTED_CANONICAL = '7ffb22a97000acd1f4448f3ac8d3829b03e99a4c4542d205eda55f10e9739f2c'
sys.path.insert(0, str(ROOT))
os.environ.update(VLM_BASE_URL='http://127.0.0.1:8082/v1', VLM_RESPONSE_FORMAT='json_object',
                  OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
from runtime import SharedGpuPool, LIBRARY, read_json, write_json, sha256
from asset_extraction import generation
from asset_extraction.quality import repair_shapes
from asset_extraction.config import load_config
from asset_extraction.rigid_publication import manifest

SLOT = asyncio.Semaphore(1)
class SinglePool(SharedGpuPool):
    async def acquire(self, stage):
        await SLOT.acquire()
        try:
            return await super().acquire(stage)
        except BaseException:
            SLOT.release()
            raise
    def release(self, gpu):
        try:
            super().release(gpu)
        finally:
            SLOT.release()
generation.GpuPool = SinglePool


def status(state, **extra):
    write_json(WORK/'status.json', dict(status=state, pid=os.getpid(), updated=time.time(), **extra))


def quarantine():
    """Preserve original records/payload, withdraw only this exact reviewed identity."""
    if (WORK/'quarantined.json').exists():
        return
    for source, name in [('completed.json', 'previous_completed.json'),
                         ('curation/curation.json', 'previous_curation.json'),
                         ('selected_generation.json', 'previous_selected_generation.json'),
                         ('technical_validity.json', 'previous_technical_validity.json')]:
        if not (WORK/name).exists():
            write_json(WORK/name, read_json(CHUNK/source))
    state = read_json(WORK/'previous_curation.json')
    selected = read_json(WORK/'previous_selected_generation.json')
    record = state['assets'][UID]
    row = next(x for x in selected['assets'] if x['obj_id'] == UID)
    if row.get('selected_attempt', 0):
        raise ValueError('Already retried: inspect current result rather than making a second retry')
    if record['entry']['canonical']['sha256'] != EXPECTED_CANONICAL:
        raise ValueError('Reviewed payload changed')
    evidence = CHUNK/'curation/asset_00002/axis_render/evidence.jpg'
    decision = dict(uid=UID, global_uid=GLOBAL_UID, reviewer='Codex visual inspection',
                    evidence=str(evidence), evidence_sha256=sha256(evidence),
                    source=str(row['image_path']), source_sha256=sha256(row['image_path']),
                    canonical_sha256=EXPECTED_CANONICAL,
                    reason='The nominal van has a short cabin-like body and lacks a plausible enclosed rear cargo/passenger section. Front/rear and side views do not form a complete commercial van. Original source is only 67x61 pixels; no exact model/length is inferred.',
                    correction='Regenerate full-body FLUX views and reconstruct once without real references; preserve source identity and all original files.')
    write_json(WORK/'visual_decision.json', decision)
    record = deepcopy(record)
    record['status'] = record['entry']['status'] = 'reject'
    record['manual_body_review'] = decision
    audit = record['entry']['quality']['audit']
    audit.update(keep=False, proportion_status='severely_mismatched', proportion_reason=decision['reason'])
    record['entry']['quality']['geometry_guard'] = dict(triggered=True,
        source='manual_body_completeness_review', reason=decision['reason'], numeric_threshold_changed=False)
    subset = deepcopy(selected)
    subset['assets'] = [row]
    subset['counts'] = dict(queries=1, generated=int(row['status']=='generated'))
    write_json(WORK/'generation.json', subset)
    write_json(WORK/'reviewed_state.json', dict(assets={UID:record}, counts=dict(queries=1, reject=1)))

    with (LIBRARY/'.rigid.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index_path = LIBRARY/'manifests/asset_library_rigid.json'
        receipts_path = LIBRARY/'manifests/rigid_publication_receipts.json'
        index = read_json(index_path)
        receipts = read_json(receipts_path)
        prior = receipts['assets'].get(GLOBAL_UID)
        central = [x for x in index['assets'] if x['uid']==GLOBAL_UID]
        if prior:
            if prior['source_sha256'] != EXPECTED_CANONICAL:
                raise ValueError('Published payload differs from visual review')
            payload = LIBRARY/'releases/rigid'/prior['entry']['asset_path']
            if sha256(payload) != prior['output_sha256']:
                raise ValueError('Published payload hash changed')
            # Leave the PLY physically intact; only remove active catalogue membership.
            write_json(WORK/'withdrawn_publication_receipt.json', prior)
        write_json(WORK/'withdrawn_central_entries.json', central)
        receipts['assets'].pop(GLOBAL_UID, None)
        remaining = [x for x in index['assets'] if x['uid'] != GLOBAL_UID]
        policy = read_json(LIBRARY/'rigid_release_policy.json')
        write_json(receipts_path, receipts)
        write_json(LIBRARY/'releases/rigid/drivelab_asset.json',
                   manifest(policy['reference'], [x['entry'] for x in receipts['assets'].values()]))
        write_json(index_path, dict(index, assets=remaining, total=len(remaining),
                                   counts=dict(Counter(x['status'] for x in remaining))))
    state['assets'][UID] = record
    state['counts'] = dict(queries=len(state['assets']), **Counter(x['status'] for x in state['assets'].values()))
    technical = read_json(CHUNK/'technical_validity.json')
    for x in technical['assets']:
        if x['obj_id']==UID:
            x['status']='reject'
            x['manual_body_review']=str(WORK/'visual_decision.json')
    receipt = read_json(CHUNK/'completed.json')
    receipt['published'] = [x for x in receipt['published'] if x['uid'] != GLOBAL_UID]
    receipt.update(counts=state['counts'], kept=sum(x['status']=='keep' for x in receipt['published']),
                   manual_body_review=dict(record=str(WORK/'visual_decision.json'), updated=time.time()))
    assert receipt['queries']==read_json(WORK/'previous_completed.json')['queries']
    write_json(CHUNK/'curation/curation.json', state)
    write_json(CHUNK/'technical_validity.json', technical)
    write_json(CHUNK/'completed.json', receipt)
    write_json(WORK/'quarantined.json', dict(uid=GLOBAL_UID, updated=time.time(),
        payload_preserved=True, query_denominator_unchanged=True))


async def main():
    with (WORK/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (WORK/'ready_for_visual_review.json').exists():
            return
        status('waiting_for_current_batch_commit')
        while not (CHUNK/'completed.json').exists():
            await asyncio.sleep(5)
        quarantine()
        status('rebuilding_full_body_without_real')
        cfg = load_config(ROOT/'configs/waymo_waymo_val_033.yaml')
        final, selected = await repair_shapes(WORK/'generation.json', read_json(WORK/'reviewed_state.json'),
            cfg, WORK/'shape_retry', renderer_python='/home/wmj/miniconda3/envs/sim/bin/python',
            gpus=[0,1,2,3,5,6])
        write_json(WORK/'ready_for_visual_review.json', dict(uid=GLOBAL_UID,
            status=final['assets'][UID]['status'], selected_attempt=selected['assets'][0].get('selected_attempt'),
            generation=str(WORK/'shape_retry/selected_generation.json'),
            curation=str(WORK/'shape_retry/curation.json'), published=False))
        status('awaiting_visual_verification')


if __name__=='__main__':
    try:
        asyncio.run(main())
    except BaseException as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
        raise
