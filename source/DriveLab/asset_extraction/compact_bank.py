"""DriveLab bank.publish implementation for explicitly migrated libraries.

Installed beside asset_extraction.bank. The policy file selects this mode;
other libraries retain the original publisher. Generation is not started here.
"""
from copy import deepcopy
from collections import Counter
from pathlib import Path
import fcntl
import hashlib
import json

from .rigid_publication import (read, atomic_json, sha256, entry_from_metadata,
                                validate_ply, copy_verified, manifest, PendingPublication)


def publish(state, library, *, dataset, scene_id):
    library = Path(library).resolve()
    policy = read(library/'rigid_release_policy.json')
    if policy.get('format') != 'drivelab_hf_rigid_v1' or policy.get('release') != 'releases/rigid':
        raise ValueError('Unsupported compact publication policy')
    release = library/'releases/rigid'
    receipts_path = library/'manifests/rigid_publication_receipts.json'
    index_path = library/'manifests/asset_library_rigid.json'
    with (library/'.rigid.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (release/'drivelab_asset.json').exists() and not receipts_path.exists():
            raise ValueError('Existing release is missing publication receipts; refusing to replace it')
        receipts = read(receipts_path) if receipts_path.exists() else {'schema_version':1, 'assets':{}}
        saved = receipts['assets']
        central = read(index_path)['assets'] if index_path.exists() else []
        index = {r.get('global_uid') or r['uid']: r for r in central}
        deleted_path = library/'manifests/deleted_assets.json'
        deleted = set(read(deleted_path).get('rigid_uids', [])) if deleted_path.exists() else set()
        added = []
        for result in state['assets'].values():
            if result['status'] not in {'keep', 'review'} or 'entry' not in result:
                continue
            row = deepcopy(result['entry'])
            if row.get('status') != result['status']:
                raise ValueError('Conflicting producer quality status')
            uid = f'{dataset}:{scene_id}:{row["uid"]}'
            if uid in deleted:
                continue
            row.update(uid=uid, global_uid=uid, dataset=dataset)
            row['provenance']['scene_id'] = scene_id
            signature = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
            if uid in saved:
                prior = saved[uid]
                if (prior.get('signature') and prior['signature'] != signature
                        or row['canonical'].get('sha256') != prior['source_sha256']):
                    raise ValueError('Existing compact publication differs: ' + uid)
                if sha256(release/prior['entry']['asset_path']) != prior['output_sha256']:
                    raise ValueError('Published compact payload changed: ' + uid)
                # Original sources/previews may have been retired; the receipt
                # and published payload suffice for an unchanged resumable job.
                if uid in index:
                    added.append(index[uid])
                    continue
                entry = prior['entry']
            else:
                source = Path(row['canonical']['canonical_path'])
                source_hash = sha256(source)
                if source_hash != row['canonical'].get('sha256'):
                    raise ValueError('Producer source hash mismatch: ' + uid)
                try:
                    entry, evidence = entry_from_metadata(row)
                    geometry = validate_ply(source, entry['size_xyz'])
                except PendingPublication as exc:
                    # Preserve review evidence without adding it to random pools.
                    token = hashlib.sha256(uid.encode()).hexdigest()[:16]
                    relative = Path('rigid/_review')/row['category']/('compact_'+token)
                    package = library/relative
                    if package.exists():
                        prior = read(package/'receipt.json')
                        if prior['signature'] != signature:
                            raise ValueError('Review package recipe differs: ' + uid)
                    digest = copy_verified(source, package/'asset.ply', expected_sha=source_hash)
                    portable = deepcopy(row)
                    portable['source_path'] = 'asset.ply'
                    portable['canonical']['canonical_path'] = 'asset.ply'
                    portable['publication_pending_reason'] = str(exc)
                    atomic_json(package/'asset.json', portable)
                    audit = row.get('quality', {}).get('audit_image')
                    if audit and Path(audit).is_file():
                        # Fixed name, independent of source machine paths.
                        copy_verified(audit, package/('review'+Path(audit).suffix))
                    atomic_json(package/'receipt.json', dict(signature=signature, sha256=digest))
                    row.update(source_path=str(relative/'asset.ply'), package_path=str(relative),
                               publication_status='pending', publication_pending_reason=str(exc))
                    row['canonical']['canonical_path'] = row['source_path']
                    index[uid] = row; added.append(row)
                    continue
                digest = copy_verified(source, release/entry['asset_path'], expected_sha=source_hash)
                saved[uid] = dict(id=uid, source_status=row['status'], source_sha256=source_hash,
                                  output_sha256=digest, signature=signature, entry=entry,
                                  evidence=evidence, geometry=geometry, disposition='published')
            row.update(source_path=str(Path('releases/rigid')/entry['asset_path']),
                       package_path='releases/rigid', publication_status='published',
                       release_manifest='releases/rigid/drivelab_asset.json')
            row['canonical']['canonical_path'] = row['source_path']
            index[uid] = row; added.append(row)
        # Receipts precede visibility; retry regenerates the manifest after an
        # interrupted commit. No existing manifest is replaced by an empty pool.
        atomic_json(receipts_path, receipts)
        atomic_json(release/'drivelab_asset.json', manifest(policy['reference'], [r['entry'] for r in saved.values()]))
        rows = sorted(index.values(), key=lambda r:r['uid'])
        atomic_json(index_path, dict(schema_version=2,total=len(rows),
                    counts=dict(Counter(r['status'] for r in rows)), assets=rows))
    return added
