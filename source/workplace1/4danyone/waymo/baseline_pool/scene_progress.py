"""Read-only evidence aggregation; never schedules preparation or generation."""
import csv
import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def snapshot():
    started = datetime.now(timezone.utc).isoformat()
    sources = []

    def read(path):
        raw = path.read_bytes()
        sources.append({'path': str(path.relative_to(ROOT)),
                        'sha256': hashlib.sha256(raw).hexdigest()})
        return json.loads(raw)

    inventory = read(ROOT / 'inventory.json')['scenes']
    rows = {}
    numbers = ['camera_tracks', 'metadata_candidate_camera_tracks', 'short_camera_tracks',
               'no_metadata_window_camera_tracks', 'primary_selected_keys',
               'primary_prepared', 'primary_running_reported', 'primary_failed_unrecovered',
               'primary_queued', 'primary_other_state', 'primary_missing_ready',
               'prepared_packages', 'alternative_packages', 'reviewed_packages',
               'input_review_candidate_packages', 'reviewed_not_passed_packages',
               'unreviewed_packages', 'prepared_official_scene_identity_keys',
               'prepared_official_unassociated_camera_keys', 'prepared_predicted_camera_keys']
    for s in inventory:
        key = (s['split'], s['scene'])
        assert key not in rows, key
        rows[key] = dict(split=key[0], scene=key[1], segment=s['segment'],
                         **dict.fromkeys(numbers, 0))
    assert len(rows) == 1000
    scan_seen = set()
    rejected_windows = defaultdict(Counter)
    for folder in ['native_scan', 'predicted_scan']:
        for f in sorted((ROOT / folder).glob('*/*.json')):
            d = read(f)
            key = (d['split'], d['scene'])
            assert key not in scan_seen, key
            scan_seen.add(key)
            row = rows[key]
            row['track_source'] = folder
            tracks = d['tracks']
            counts = Counter(t['state'] for t in tracks)
            assert set(counts) <= {'metadata_candidate_not_visibility_accepted',
                                   'short_track', 'no_metadata_window'}, counts
            row['camera_tracks'] = len(tracks)
            row['metadata_candidate_camera_tracks'] = counts['metadata_candidate_not_visibility_accepted']
            row['short_camera_tracks'] = counts['short_track']
            row['no_metadata_window_camera_tracks'] = counts['no_metadata_window']
            for t in tracks:
                rejected_windows[key].update(t.get('rejected_window_reasons', {}))
    assert scan_seen == set(rows)

    # Read each controller once; states below are reports, not process-liveness claims.
    status = {name: read(ROOT / name).get('jobs', {}) for name in
              ['native_prepare_status.json', 'predicted_prepare_status.json',
               'pilot_prepare_status.json', 'processed_pilot_status.json']}
    candidates = []
    for queue, controller in [('native_queue.json', 'native_prepare_status.json'),
                              ('predicted_queue.json', 'predicted_prepare_status.json'),
                              ('pilot_manifest.json', 'pilot_prepare_status.json'),
                              ('processed_pilot_queue.json', 'processed_pilot_status.json')]:
        candidates.extend((c, controller) for c in read(ROOT / queue)['candidates'])
    assert len(candidates) == 5364
    candidate_keys = [(c['split'], c['scene'], c['identity_key']) for c, _ in candidates]
    assert len(set(candidate_keys)) == len(candidate_keys)

    ready_paths = sorted([*(ROOT / 'samples').glob('*/*/*/READY.json'),
                          *(ROOT / 'retry').glob('*/*/*/*/READY.json')])
    ready_set = set(ready_paths)
    identities = defaultdict(lambda: defaultdict(set))
    for f in ready_paths:
        d = read(f)
        c = d['candidate']
        key = (c['split'], c['scene'])
        row = rows[key]
        row['prepared_packages'] += 1
        row['alternative_packages'] += int(f.relative_to(ROOT).parts[0] == 'retry')
        scope = c['identity_scope']
        assert scope in {'official_2d3d_association', 'official_camera_local_only',
                         'predicted_camera_local_not_official'}, scope
        identities[key][scope].add(c['identity_key'])
        review = f.parent / 'review.json'
        if review.exists():
            r = read(review)
            row['reviewed_packages'] += 1
            row['input_review_candidate_packages'] += int(r.get('input_review_passed') is True)
            row['reviewed_not_passed_packages'] += int(r.get('input_review_passed') is not True)
        else:
            row['unreviewed_packages'] += 1

    for c, controller in candidates:
        row = rows[c['split'], c['scene']]
        row['primary_selected_keys'] += 1
        ready = Path(c['sample_root']).parent / 'READY.json'
        state = status[controller].get(c['review_id'], {}).get('state')
        if ready in ready_set:
            category = 'primary_prepared'
        elif state == 'running':
            category = 'primary_running_reported'
        elif state == 'preparation_failed_not_identity_rejected':
            category = 'primary_failed_unrecovered'
        elif state == 'prepared_review_pending':
            category = 'primary_missing_ready'
        elif state is None:
            category = 'primary_queued'
        else:
            category = 'primary_other_state'
        row[category] += 1

    for key, row in rows.items():
        for scope, column in [('official_2d3d_association', 'prepared_official_scene_identity_keys'),
                              ('official_camera_local_only', 'prepared_official_unassociated_camera_keys'),
                              ('predicted_camera_local_not_official', 'prepared_predicted_camera_keys')]:
            row[column] = len(identities[key][scope])
        row['rejected_window_reason_counts'] = dict(sorted(rejected_windows[key].items()))
        row['quality_accepted_people'] = None
        assert row['primary_selected_keys'] == sum(row[c] for c in numbers if c.startswith('primary_') and c != 'primary_selected_keys')
        assert row['prepared_packages'] == row['reviewed_packages'] + row['unreviewed_packages']
    totals = {c: sum(r[c] for r in rows.values()) for c in numbers}
    assert totals['prepared_packages'] == len(ready_paths)
    assert totals['primary_prepared'] + totals['alternative_packages'] == len(ready_paths)
    assert totals['camera_tracks'] == 40318
    ordered = [rows[k] for k in sorted(rows)]
    result = {'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
              'scene_count': len(rows), 'totals': totals, 'rows': ordered, 'sources': sources,
              'definitions': {
                  'primary_selected_keys': 'Official scene identity when associated; otherwise unresolved camera track. Not distinct people.',
                  'prepared_packages': 'READY files captured in this snapshot; not quality accepted.',
                  'primary_failed_unrecovered': 'Primary window has failure report and no captured READY. Another window of this identity may succeed.',
                  'rejected_window_reason_counts': 'Window-level reason counts; overlapping windows and multiple reasons are nonexclusive. Not rejected people.',
                  'reviewed_not_passed_packages': 'Includes retained completion candidates and uncertainty; not automatic rejection.',
                  'primary_running_reported': 'Controller snapshot only; liveness checked separately.',
                  'quality_accepted_people': 'Unknown pending quality review and unresolved cross-camera identities.',
                  'snapshot': 'Sources read during stated interval; not an atomic view of live queues. Hashes identify exact bytes read.'}}
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(ordered[0]))
    writer.writeheader()
    for row in ordered:
        writer.writerow({**row, 'rejected_window_reason_counts': json.dumps(row['rejected_window_reason_counts'], sort_keys=True)})
    for name, content in [('scene_progress.json', json.dumps(result, ensure_ascii=False, indent=2) + '\n'),
                          ('scene_progress.csv', buffer.getvalue())]:
        target = ROOT / name
        temp = target.with_suffix(target.suffix + f'.{os.getpid()}.tmp')
        temp.write_text(content)
        temp.replace(target)
    print(json.dumps({'scenes': len(rows), 'totals': totals}, ensure_ascii=False))


if __name__ == '__main__':
    snapshot()
