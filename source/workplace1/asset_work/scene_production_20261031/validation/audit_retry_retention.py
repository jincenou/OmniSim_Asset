"""Read-only audit of selected shape retries and their preserved source inputs."""
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGNS = [ROOT, ROOT.parent / 'web_vehicles_20261103']


def load(path):
    return json.loads(path.read_text())


def check_file(path, expected=None):
    p = Path(path)
    result = {'path': str(p), 'exists': p.is_file()}
    if p.is_file():
        result['bytes'] = p.stat().st_size
        if expected:
            result['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
            result['matches_recorded_hash'] = result['sha256'] == expected
    return result


def main():
    records = []
    for campaign in CAMPAIGNS:
        for selected_path in sorted((campaign / 'chunks').glob('*/selected_generation.json')):
            chunk = selected_path.parent
            selected = load(selected_path)
            retries = [x for x in selected['assets'] if x.get('selected_attempt')]
            if not retries:
                continue
            original = {x['obj_id']: x for x in load(chunk / 'generation/generation.json')['assets']}
            for row in retries:
                uid = row['obj_id']
                before = original[uid]
                h = row.get('stages', {}).get(row['route'], {})
                flux = row.get('stages', {}).get('flux', {}).get('images', [])
                expected_inputs = [x['path'] for x in flux]
                checks = {
                    'same_identity': row['obj_id'] == before['obj_id'],
                    'same_category': row['category'] == before['category'],
                    'no_real_references': row.get('real_images') == [],
                    'actual_reconstruction_real_count_zero': h.get('real_count') == 0,
                    'actual_reconstruction_inputs_exactly_flux': bool(expected_inputs) and h.get('input_images') == expected_inputs,
                }
                files = [check_file(before['image_path'], before.get('image_sha256'))]
                files += [check_file(x['rgba_path'], x.get('sha256')) for x in before.get('real_images', [])]
                files += [check_file(x['path'], x.get('sha256')) for x in before.get('stages', {}).get('flux', {}).get('images', [])]
                files += [check_file(x['path'], x.get('sha256')) for x in flux]
                checks['input_files_present_and_hashes_match'] = all(
                    x['exists'] and x.get('bytes', 0) > 0 and x.get('matches_recorded_hash', True)
                    for x in files)
                # Raw payloads can be large: existence/size only, explicitly not a rehash.
                raw_files = []
                for attempt in [before, row]:
                    if attempt.get('raw_path'):
                        raw_files.append(check_file(attempt['raw_path']))
                checks['both_raw_payloads_present'] = len(raw_files) == 2 and all(
                    x['exists'] and x.get('bytes', 0) > 0 for x in raw_files)
                removed = {}
                for cleanup in chunk.glob('rejected_output_cleanup_*.json'):
                    for path in load(cleanup).get('outputs_removed', []):
                        removed[path] = str(cleanup)
                for file in raw_files:
                    if not file['exists'] and file['path'] in removed:
                        file['documented_prior_cleanup'] = removed[file['path']]
                records.append({'campaign': campaign.name, 'chunk': chunk.name,
                                'obj_id': uid, 'route': row['route'], 'checks': checks,
                                'files': files, 'raw_files': raw_files,
                                'passed': all(checks.values())})
    report = {'checked_at': time.time(), 'selected_retries': len(records),
              'passed': all(x['passed'] for x in records), 'records': records,
              'input_and_no_real_checks_passed': all(all(v for k, v in x['checks'].items()
                    if k != 'both_raw_payloads_present') for x in records),
              'missing_raw_payloads': sum(not f['exists'] for x in records for f in x['raw_files']),
              'missing_raw_with_cleanup_receipt': sum('documented_prior_cleanup' in f
                    for x in records for f in x['raw_files']),
              'scope': 'Selected completed H/B0 retry inputs: actual stage input list, source/FLUX hashes, original and retry raw file presence. Does not audit unselected/unfinished retries, full raw payload hashes, or visual quality.'}
    destination = ROOT / 'validation/retry_retention_audit_20261103.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({'report': str(destination), 'selected_retries': len(records),
                      'passed': report['passed'],
                      'input_and_no_real_checks_passed': report['input_and_no_real_checks_passed'],
                      'missing_raw_payloads': report['missing_raw_payloads'],
                      'missing_raw_with_cleanup_receipt': report['missing_raw_with_cleanup_receipt']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
