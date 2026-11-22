"""Prioritize manual input review using existing evidence, never reject/schedule."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    rows = []
    for path in sorted([*(ROOT/'samples').glob('*/*/*/READY.json'),
                        *(ROOT/'retry').glob('*/*/*/*/READY.json')]):
        raw = path.read_bytes()
        d = json.loads(raw)
        c = d['candidate']
        area = d.get('mask_risks', {}).get('mask_area_per_frame')
        zeros = [] if area is None else [i for i, v in enumerate(area) if v == 0]
        if area is not None:
            assert len(area) == c['output_slots'], path
        longest = current = 0
        for value in area or []:
            current = current+1 if value == 0 else 0
            longest = max(longest, current)
        audit_path = path.parent/'skeleton_audit.json'
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        review_path = path.parent/'review.json'
        review = json.loads(review_path.read_text()) if review_path.exists() else {}
        rows.append({'package': str(path.parent), 'split': c['split'], 'scene': c['scene'],
                     'review_id': c['review_id'], 'identity_scope': c['identity_scope'],
                     'frames': c['output_slots'], 'mask_evidence_present': area is not None,
                     'empty_binary_mask_frames': len(zeros) if area is not None else None,
                     'empty_binary_mask_fraction': len(zeros)/len(area) if area else None,
                     'longest_empty_binary_mask_frames': longest if area is not None else None,
                     'median_confident_keypoints': audit.get('reprojection', {}).get('confident_points_per_frame_min_median', [None,None])[1],
                     'bone_numeric_flags': audit.get('flags'),
                     'visual_review_state': review.get('state','unreviewed'),
                     'input_review_passed': review.get('input_review_passed'),
                     'ready_sha256': hashlib.sha256(raw).hexdigest()})
    rows.sort(key=lambda r: (-(r['empty_binary_mask_fraction'] or 0),
                            -(r['longest_empty_binary_mask_frames'] or 0),r['package']))
    summary = {'prepared_packages': len(rows),
               'mask_evidence_unknown': sum(not r['mask_evidence_present'] for r in rows),
               'any_empty_binary_mask': sum((r['empty_binary_mask_frames'] or 0)>0 for r in rows),
               'at_least_half_frames_empty_binary_mask': sum((r['empty_binary_mask_fraction'] or 0)>=0.5 for r in rows),
               'at_least_half_empty_without_bone_numeric_flags': sum((r['empty_binary_mask_fraction'] or 0)>=0.5 and r['bone_numeric_flags']==[] for r in rows),
               'review_states': dict(Counter(r['visual_review_state'] for r in rows))}
    payload = {'utc': datetime.now(timezone.utc).isoformat(), 'summary':summary, 'rows':rows,
               'limits':['Counts refer to packages, not distinct people.',
                         'Empty means no stored mask pixels >127/255, as recorded by the existing gate; soft alpha may still exist.',
                         'Half-frame bin is descriptive, not an admission/rejection threshold.',
                         'No automatic exclusion or change to the frozen pipeline.',
                         'No numeric flag does not validate pose or identity.',
                         'Per-frame counts reused from READY evidence; not all mask archives independently decoded in this sweep.']}
    (ROOT/'input_risk_inventory.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    with (ROOT/'input_risk_inventory.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for row in rows:
            w.writerow({**row,'bone_numeric_flags':json.dumps(row['bone_numeric_flags'])})
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    main()
