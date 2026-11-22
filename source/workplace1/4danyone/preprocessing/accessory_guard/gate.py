"""Hash-bound source/accessory review required by new project generation runs.

This validates an explicit review, not an automatic semantic correctness claim.
The official repository and already-frozen experiments remain unchanged.
"""
from pathlib import Path
import hashlib
import json

SCHEMA = 'accessory-input-review-v1'

class ReviewRequired(RuntimeError):
    pass

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def check_file(row):
    p = Path(row['path'])
    if not p.is_file() or sha(p) != row['sha256']:
        raise ReviewRequired(f'Missing or changed reviewed file: {p}')
    return p.resolve()

def require_review(path, input_video):
    path = Path(path)
    try:
        review = json.loads(path.read_text())
        if review['schema'] != SCHEMA or review['decision'] != 'approved_for_generation':
            raise ReviewRequired('Accessory review missing approval; do not generate')
        if not isinstance(review['reviewer'], str) or not review['reviewer'].strip():
            raise ReviewRequired('Explicit reviewer required')
        if review['accessories'] not in ('none_observed_after_review', 'preserved', 'repaired'):
            raise ReviewRequired('Unknown or missing accessory: repair/review first')
        n = review['frames']
        if type(n) is not int or n <= 0:
            raise ReviewRequired('Invalid frame count')
        # Cover all input frames; a sampled W02/W04 pilot is not full approval.
        if sorted(set(review['reviewed_frames'])) != list(range(n)):
            raise ReviewRequired('Incomplete frame review; exploratory trial only')
        for field in ('source_input_compared', 'no_new_bystander_leakage',
                      'no_visible_accessory_whiteout', 'occlusion_drift_checked'):
            if review[field] is not True:
                raise ReviewRequired(f'Unresolved review item: {field}')
        bound = review['files']
        source = check_file(bound['source_rgb'])
        mask = check_file(bound['mask'])
        actual = check_file(bound['generation_input'])
        if actual != Path(input_video).resolve():
            raise ReviewRequired('Review belongs to a different generation input')
        if source == actual or mask == actual:
            raise ReviewRequired('Need original RGB and separate mask evidence')
        if not review['evidence']:
            raise ReviewRequired('Missing source/mask/composite comparison evidence')
        tuple(map(check_file, review['evidence']))
        # Optional cached conditioning must also be bound; caller checks its path.
        if 'conditioning_source' in bound:
            check_file(bound['conditioning_source'])
        return review
    except ReviewRequired:
        raise
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ReviewRequired(f'Accessory input review required at {path}: {error}') from error

def check_root(root):
    root = Path(root)
    review = require_review(root/'preflight/accessory_review.json', root/'inputs/main.mp4')
    meta = root/'preflight/conditioning/metadata.json'
    if meta.exists():
        source = (meta.parent/json.loads(meta.read_text())['source_video']).resolve()
        bound = review['files'].get('conditioning_source')
        if bound is None or check_file(bound) != source:
            raise ReviewRequired('Cached conditioning appearance is not covered by this review')
    return review
