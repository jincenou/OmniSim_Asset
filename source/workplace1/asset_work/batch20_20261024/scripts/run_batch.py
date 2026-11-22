"""Run every frozen query in bounded chunks and publish one central asset bank."""
from __future__ import annotations
import asyncio
from collections import Counter, defaultdict, deque
from copy import deepcopy
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

P = Path('/mnt/ojc/pre_flux_pipeline')
SOURCE_BATCH = Path(__file__).resolve().parents[1]
B = Path(os.environ.get('ASSET_BATCH_ROOT', str(SOURCE_BATCH)))
BATCH_ID = SOURCE_BATCH.name if B == SOURCE_BATCH else f'{SOURCE_BATCH.name}:{B.name}'
LIBRARY = Path('/mnt/ojc/workplace1/asset_library')
sys.path.insert(0, str(P))
from asset_extraction.config import load_config
from asset_extraction.generation import generate, prepare_jobs, sha256, digest, checked_outputs
from asset_extraction.export import curate, export_library
from asset_extraction.io_utils import read_json, write_json

GPUS = [int(value) for value in os.environ.get('ASSET_GENERATION_GPUS','3,4,5,6').split(',')]
CHUNK_SIZE = 40
RUN_SUFFIX = os.environ.get('ASSET_RUN_SUFFIX', '10')
CONFIG_SUFFIX = os.environ.get('ASSET_CONFIG_SUFFIX', '')
RENDERER = '/home/wmj/miniconda3/envs/sim/bin/python'


def publish(chunk_library, dataset, chunk_name, destination=LIBRARY):
    """Hardlink verified payloads, atomically extend the authoritative JSON index."""
    destination.mkdir(parents=True, exist_ok=True)
    manifests = destination / 'manifests'
    manifests.mkdir(exist_ok=True)
    source_manifest = Path(chunk_library) / 'manifests/asset_library_rigid.json'
    entries = deepcopy(read_json(source_manifest)['assets'])
    with (manifests / '.publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main = manifests / 'asset_library_rigid.json'
        rows = read_json(main)['assets'] if main.exists() else []
        by_uid = {r['uid']: r for r in rows}
        if len(by_uid) != len(rows):
            raise ValueError('Central library has duplicate identities')
        published = []
        for row in entries:
            original_id = row['uid']
            row['uid'] = f'{BATCH_ID}:{dataset}:{original_id}'
            row['global_uid'] = row['uid']
            row['dataset'] = dataset
            row['provenance'].update(batch_id=BATCH_ID, batch_chunk=chunk_name,
                                     source_obj_id=original_id)
            relative = Path(row['canonical']['canonical_path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Publication requires a relative payload path')
            source, target = Path(chunk_library) / relative, destination / relative
            expected = row['canonical']['sha256']
            if sha256(source) != expected:
                raise ValueError('Chunk payload changed')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(source, target)
            if sha256(target) != expected:
                raise ValueError('Central payload hash mismatch')
            previous = by_uid.get(row['uid'])
            if previous is not None and previous != row:
                raise ValueError('Publication would replace an existing logical asset')
            if previous is None:
                rows.append(row)
                by_uid[row['uid']] = row
            published.append(dict(uid=row['uid'], canonical_path=str(target), sha256=expected,
                                  source_obj_id=original_id, status=row['status']))
        physical = {}
        for row in rows:
            canonical = row['canonical']
            key = canonical['sha256']
            physical[key] = dict(canonical_path=canonical['canonical_path'], category=row['category'],
                                 format='ply', sha256=key,
                                 size=(destination / canonical['canonical_path']).stat().st_size,
                                 tier='candidate')
        index = manifests / 'global_asset_manifest.jsonl'
        temporary = index.with_suffix('.jsonl.tmp')
        temporary.write_text(''.join(json.dumps(v, ensure_ascii=False, allow_nan=False) + '\n'
                                    for _, v in sorted(physical.items())))
        temporary.replace(index)
        write_json(main, dict(schema_version=2, total=len(rows),
                              counts=dict(Counter(r['category'] for r in rows)), assets=rows))
        return published


def prune_work(chunk, published):
    """Remove only this chunk's intermediates after hashed central publication."""
    ledger = chunk / 'pruning.json'
    if ledger.exists() and read_json(ledger).get('status') == 'completed':
        return
    for row in published:
        if sha256(row['canonical_path']) != row['sha256']:
            raise ValueError('Cannot prune before verifying the published payload')
    state = read_json(chunk / 'generation/generation.json')
    ids = {r['source_obj_id'] for r in published}
    paths = []
    for row in state['assets']:
        if row['obj_id'] not in ids:
            continue  # Keep failed assets' original evidence for diagnosis.
        index = row['item_index']
        curation_roots = [chunk / 'curation', *chunk.glob('curation_retry_*'),
                          *chunk.glob('service_outage_*/curation')]
        for root in curation_roots:
            paths.extend((root / f'asset_{index:05d}').glob('*.ply'))
        paths.append(Path(row['raw_path']))
    for library in chunk.glob('library*'):
        paths.extend((library / 'payloads').rglob('*.ply'))
    records = read_json(ledger)['files'] if ledger.exists() else [
        dict(path=str(p), sha256=sha256(p), bytes=p.stat().st_size) for p in paths if p.is_file()]
    write_json(ledger, dict(status='pruning', files=records, replacement_payloads=published,
                           note='Intermediate geometry removed after publication; logs, transforms and hashes retained.'))
    for record in records:
        path = Path(record['path'])
        if not path.is_relative_to(chunk) or path.suffix != '.ply':
            raise ValueError('Refusing to remove a path outside the current chunk')
        if path.exists():
            if sha256(path) != record['sha256']:
                raise ValueError('Intermediate changed during pruning')
            path.unlink()
    write_json(ledger, dict(status='completed', files=records, replacement_payloads=published,
                           intermediate_bytes_unlinked=sum(r['bytes'] for r in records)))


async def audit_chunk(chunk, config):
    """Retry infrastructure failures only; preserve every completed quality decision."""
    state_path = chunk / 'curation/curation.json'
    result = read_json(state_path) if state_path.exists() else None
    if result is None or 'counts' not in result:
        result = await curate(chunk / 'generation/generation.json', chunk / 'library', config,
                              renderer_python=RENDERER, gpus=GPUS, work_dir=chunk / 'curation')
    def unavailable(state):
        return [uid for uid,item in state['assets'].items() if item.get('status')=='failed'
                and any(name in item.get('error','') for name in
                    ('APIConnectionError','APITimeoutError','InternalServerError','RateLimitError'))]
    failed = unavailable(result)
    if failed and not result.get('service_retry'):
        subset = deepcopy(read_json(chunk / 'generation/generation.json'))
        subset['assets'] = [r for r in subset['assets'] if r['obj_id'] in failed]
        retry_generation = write_json(chunk / 'retry_generation_01.json', subset)
        retry_work, retry_library = chunk / 'curation_retry_01', chunk / 'library_retry_01'
        repaired = await curate(retry_generation,retry_library,config,renderer_python=RENDERER,
                                gpus=GPUS,work_dir=retry_work)
        write_json(chunk / 'curation_before_retry.json', result)
        result['assets'].update(repaired['assets'])
        result['service_retry'] = dict(obj_ids=failed, signature=repaired['signature'],
                                       state=str(retry_work/'curation.json'))
        result['counts'] = dict(queries=len(result['assets']),
                               **dict(Counter(r['status'] for r in result['assets'].values())))
        entries = [r['entry'] for r in result['assets'].values() if 'entry' in r]
        result['status'] = 'curated' if len(entries)==len(result['assets']) else 'incomplete'
        if entries:
            if (chunk / 'library').exists():
                (chunk / 'library').rename(chunk / 'library_before_retry')
            result['library'] = str(export_library(entries,chunk/'library'))
        write_json(state_path,result)
    if unavailable(result):
        write_json(chunk / 'service_errors.json', unavailable(result))
        raise RuntimeError(f'Quality service is unavailable for {len(unavailable(result))} assets; '
                           f'raw outputs and all completed decisions are preserved: {chunk}')
    return result


async def process_dataset(dataset):
    run = B / 'runs' / f'{dataset}{RUN_SUFFIX}'
    handoff, routes = run / 'handoff/flux_input.json', run / 'handoff/asset_routes.json'
    rows, routing = read_json(handoff), read_json(routes)
    if sha256(handoff) != routing['flux_input_sha256']:
        raise ValueError('Routes do not match the complete query registry')
    # Expose all fixed scenes early, then continue through every admitted query.
    # Within each scene, cycle categories instead of generating every rail first.
    grouped=defaultdict(lambda:defaultdict(list))
    for row in sorted(rows,key=lambda r:r['obj_id']):
        grouped[row['scene_id']][row['category']].append(row)
    def interleave(groups):
        queues=[deque(values) for _,values in sorted(groups.items())]
        return [queue.popleft() for _ in range(max(map(len,queues),default=0)) for queue in queues if queue]
    rows=interleave({scene:interleave(categories) for scene,categories in grouped.items()})
    frozen = B / 'evaluation' / f'{dataset}_queries.json'
    registry = dict(dataset=dataset, queries=len(rows), handoff=str(handoff),
                    handoff_sha256=sha256(handoff), routing_sha256=sha256(routes),
                    obj_ids=[r['obj_id'] for r in rows], chunk_size=CHUNK_SIZE,
                    order='round_robin_scenes_then_categories_within_scene', flux_seed='chunk_local_index*1000 + view_index')
    if frozen.exists() and read_json(frozen) != registry:
        raise ValueError('Frozen queries changed after generation began')
    write_json(frozen, registry)
    route_by_id = {r['obj_id']: r for r in routing['assets']}
    config = load_config(B / 'configs' / f'{dataset}{CONFIG_SUFFIX}.yaml')
    config.vlm.concurrency = 4
    generation_config = read_json(P / 'configs/generation.example.json')
    for start in range(0, len(rows), CHUNK_SIZE):
        name = f'{dataset}_{start // CHUNK_SIZE:03d}'
        chunk = B / 'chunks' / name
        chunk.mkdir(parents=True, exist_ok=True)
        completion = chunk / 'completed.json'
        if completion.exists():
            done = read_json(completion)
            for item in done['published']:
                if sha256(item['canonical_path']) != item['sha256']:
                    raise ValueError('Published chunk payload changed')
            prune_work(chunk, done['published'])
            print(name, 'verified completed', flush=True)
            yield done
            continue
        if shutil.disk_usage(B).free < 12 * 1024**3:
            raise OSError('Less than 12 GiB free: preserve outputs and release space before continuing')
        selected = rows[start:start + CHUNK_SIZE]
        selected_input = write_json(chunk / 'flux_input.json', selected)
        selected_routes = dict(routing, flux_input=str(selected_input),
            flux_input_sha256=sha256(selected_input), assets=[route_by_id[r['obj_id']] for r in selected])
        selected_route_path = write_json(chunk / 'asset_routes.json', selected_routes)
        if not (chunk / 'published.json').exists():
            bundle = prepare_jobs(selected_input, selected_route_path, generation_config)
            print(name, 'generating', len(selected), flush=True)
            await generate(bundle, chunk / 'generation', GPUS)
            # Ordinary generation failures stay in the cohort; curate every valid result.
            print(name, 'curating', flush=True)
            await audit_chunk(chunk,config)
            published = publish(chunk / 'library', dataset, name) if (chunk / 'library').exists() else []
            write_json(chunk / 'published.json', published)
        else:
            published = read_json(chunk / 'published.json')
        generated = read_json(chunk / 'generation/generation.json')
        curated = read_json(chunk / 'curation/curation.json')
        done = dict(dataset=dataset, chunk=name, queries=len(selected),
                    generation_counts=generated['counts'], curation_counts=curated['counts'],
                    generation_sha256=sha256(chunk / 'generation/generation.json'),
                    curation_sha256=sha256(chunk / 'curation/curation.json'), published=published)
        write_json(completion, done)
        prune_work(chunk, published)
        print(name, 'published', len(published), curated['counts'], flush=True)
        yield done
    write_json(B / 'reports' / f'{dataset}_generation_complete.json',
               dict(queries=len(rows), chunks=(len(rows) + CHUNK_SIZE - 1) // CHUNK_SIZE))


async def main():
    pending = ['nuscenes', 'waymo']
    streams={}
    completed_chunks=Counter()
    while pending:
        ready = [d for d in pending if (B / 'runs' / f'{d}{RUN_SUFFIX}/handoff/asset_routes.json').is_file()]
        if not ready:
            for dataset in pending:
                status = B / 'runs' / f'{dataset}{RUN_SUFFIX}/status.asset_extraction.json'
                if status.exists() and read_json(status).get('status') == 'failed':
                    raise RuntimeError(f'{dataset} frontend failed: {status}')
            write_json(B / 'batch_status.json', dict(stage='waiting_for_frontend', pending=pending, time=time.time()))
            await asyncio.sleep(10)
            continue
        dataset = min(ready,key=lambda d:completed_chunks[d])
        write_json(B / 'batch_status.json', dict(stage='generation_and_curation', dataset=dataset, time=time.time()))
        if dataset not in streams:
            streams[dataset]=process_dataset(dataset)
        try:
            await streams[dataset].__anext__()
            completed_chunks[dataset]+=1
        except StopAsyncIteration:
            pending.remove(dataset)
    write_json(B / 'batch_status.json', dict(stage='generation_complete_metrics_pending', time=time.time()))


async def recover():
    """Recover failed queries in separate, resumable attempts; keep original receipts intact."""
    plan_path = B / 'recovery_plan.json'
    if plan_path.exists():
        plan = read_json(plan_path)
    else:
        chunks = []
        for chunk in sorted((B / 'chunks').iterdir()):
            path = chunk / 'generation/generation.json'
            if not path.exists():
                continue
            state = read_json(path)
            ids = [row['obj_id'] for row in state['assets'] if row['status'] == 'failed']
            if ids:
                chunks.append(dict(chunk=chunk.name, obj_ids=ids, generation_sha256=sha256(path),
                    reusable_flux=sum(row['obj_id'] in ids and row['stages'].get('flux',{}).get('status')=='succeeded' for row in state['assets'])))
        chunks.sort(key=lambda item: (-item['reusable_flux'], item['chunk']))
        plan = dict(attempt='recovery_01', chunks=chunks, queries=sum(len(c['obj_ids']) for c in chunks),
                    policy='Only initial generation failures; original inputs, seeds, routes and quality thresholds retained')
        write_json(plan_path, plan)
    write_json(B / 'batch_status.json', dict(stage='recovery_running', time=time.time(), queries=plan['queries']))
    for item in plan['chunks']:
        chunk = B / 'chunks' / item['chunk']
        work = chunk / plan['attempt']
        completion = work / 'completed.json'
        if completion.exists():
            done = read_json(completion)
            if done['generation_sha256'] != sha256(work/'generation/generation.json') or done['curation_sha256'] != sha256(work/'curation/curation.json'):
                raise ValueError('Recovery receipts changed')
            if any(sha256(row['canonical_path']) != row['sha256'] for row in done['published']):
                raise ValueError('Recovery publication changed')
            continue
        original_path = chunk / 'generation/generation.json'
        if sha256(original_path) != item['generation_sha256']:
            raise ValueError('Initial attempt changed during recovery')
        original = read_json(original_path)
        old_by_id = {row['obj_id']: row for row in original['assets']}
        bundle = prepare_jobs(chunk/'flux_input.json', chunk/'asset_routes.json', original['config'])
        selected = [dict(row, item_index=index, flux_seed_index=row['item_index'])
                    for index, row in enumerate(row for row in bundle['assets'] if row['obj_id'] in item['obj_ids'])]
        bundle.update(assets=selected, recovery_parent=str(original_path), recovery_parent_sha256=item['generation_sha256'])
        bundle['signature'] = digest(dict(parent=bundle['signature'], assets=selected, parent_sha256=item['generation_sha256']))
        state_path = work / 'generation/generation.json'
        if not state_path.exists():
            assets = []
            for row in selected:
                previous = old_by_id[row['obj_id']]['stages'].get('flux', {})
                stages = {}
                if previous.get('status') == 'succeeded':
                    checked_outputs(previous)
                    stages['flux'] = dict(previous, item_index=row['item_index'])
                assets.append(dict(row, stages=stages, status='pending'))
            write_json(state_path, dict(schema_version=1, signature=bundle['signature'], config=bundle['config'], assets=assets))
        print(item['chunk'], 'recovering', len(selected), 'GPUs', GPUS, flush=True)
        await generate(bundle, work/'generation', GPUS)
        dataset = item['chunk'].split('_')[0]
        config = load_config(B/'configs'/f'{dataset}{CONFIG_SUFFIX}.yaml')
        config.vlm.concurrency = 4
        await audit_chunk(work, config)
        published = publish(work/'library', dataset, f'{chunk.name}/{plan["attempt"]}') if (work/'library').exists() else []
        write_json(work/'published.json', published)
        generated = read_json(state_path)
        curated = read_json(work/'curation/curation.json')
        write_json(completion, dict(queries=len(selected), generation_counts=generated['counts'],
            curation_counts=curated['counts'], generation_sha256=sha256(state_path),
            curation_sha256=sha256(work/'curation/curation.json'), published=published))
        print(item['chunk'], 'recovered', len(published), curated['counts'], flush=True)
    write_json(B/'batch_status.json', dict(stage='recovery_finished', time=time.time(), queries=plan['queries']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--recover', action='store_true'); args = parser.parse_args()
    with (B / '.batch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            asyncio.run(recover() if args.recover else main())
        except Exception as exc:
            write_json(B / 'batch_status.json', dict(stage='failed', error=f'{type(exc).__name__}: {exc}', time=time.time()))
            raise
