"""Assign saved observational training records to conservative reference cores.

This is post-training analysis only; no heldout content, parameter update or
retroactive reconstruction of unrecorded gradient history.
"""
import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from prepare_diagnostics import ROOT, BASE, PARTS, COLORS, project, conservative
from render_baseline import sha


def write_csv(path, rows):
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--label', required=True); args = parser.parse_args()
    assert args.label in ['N04', 'N11']; cv2.setNumThreads(2)
    label = args.label; run = ROOT / 'runs' / f'{label}_33k_v1'; sample = BASE / 'samples' / label
    data = json.loads((run / 'asset_camera_data.json').read_text())
    source = json.loads((BASE / 'four_dg/data' / label / 'fdanyone.json').read_text())
    trace = [json.loads(line) for line in (run / 'trace.jsonl').read_text().splitlines()]
    saved_exposure = json.loads((run / 'sample_exposure.json').read_text())
    counts = {}; update_counts = {}
    for row in trace:
        key = (row['view'], row['frame']); counts[key] = counts.get(key, 0) + 1
        update_counts[key] = update_counts.get(key, 0) + int(row['updated'])
    assert all(view in data['train_views'] for view, _ in counts)
    assert {f'{v}:{t}': n for (v, t), n in counts.items()} == saved_exposure
    points = np.load(sample / 'generation/preflight/conditioning/keypoints_3d.npy')
    cams = json.loads((sample / 'generation/outputs/turbo/cameras.json').read_text())['cameras']
    out = ROOT / 'audit/training' / label; out.mkdir(parents=True, exist_ok=False)
    @lru_cache(maxsize=128)
    def core(view, frame):
        assert view in data['train_views'], 'No heldout observation in training diagnostic'
        c = next(c for c in cams if c['camera_id'] == view)
        k = np.asarray(c['K'], float); k[:2] *= .5
        uv, _ = project(points[frame], c['camera_to_world'], k)
        rgba = np.asarray(Image.open(sample / 'data/images' / f'cam{view:02d}' / f'{frame:06d}.png'), np.float32) / 255
        roi, support = conservative(uv, rgba[..., 3] > .5, label, view)
        return roi, support
    exposure = []
    for (view, frame), n in sorted(counts.items()):
        roi, support = core(view, frame)
        for p, part in enumerate(PARTS, 1):
            area = int((roi == p).sum())
            exposure.append(dict(label=label, view=view, frame=frame, part=part, reference_core_pixels=area,
                training_renders=n, optimizer_updates=update_counts[view, frame], reference_core_pixel_renders=area * n,
                reference_core_pixel_updates=area * update_counts[view, frame], centerline_support=support[p - 1],
                status='estimated_reference_core' if area else 'unknown'))
    write_csv(out / 'exposure.csv', exposure)
    rows = []
    files = sorted((run / 'diagnostics').glob('*.npz'))
    for path in files:
        with np.load(path) as z:
            view, frame = int(z['view']), int(z['frame']); roi, _ = core(view, frame)
            uv = z['uv']; n = len(uv)
            cam = next(c for c in data['cameras'] if c['view'] == view)
            w = np.linalg.inv(np.asarray(cam['camtoworld']))
            depths = (z['world_xyz'] @ w[:3, :3].T + w[:3, 3])[:, 2]
            valid = np.isfinite(uv).all(1) & (depths > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < 352) & (uv[:, 1] >= 0) & (uv[:, 1] < 640)
            kind = 'gradient' if path.name.startswith('gradient') else 'append' if path.stem.endswith('append') else 'prune'
            if kind == 'gradient': valid &= z['radii'] > 0
            ids = np.where(valid)[0]
            x = np.clip(np.rint(uv[ids, 0]).astype(int), 0, 351); y = np.clip(np.rint(uv[ids, 1]).astype(int), 0, 639)
            groups = np.zeros(n, np.uint8); groups[ids] = roi[y, x]
            grad = np.linalg.norm(z['screen_gradient'][:, :2], axis=1) if kind == 'gradient' else None
            for p, part in enumerate(['unknown'] + PARTS):
                sel = groups == p; area = int((roi == p).sum()) if p else None
                count = int(sel.sum())
                rows.append(dict(label=label, kind=kind, iteration=int(path.stem.split('_')[1]), view=view, frame=frame, part=part,
                    reference_core_pixels=area, point_count=count, points_per1000_reference_pixels=count * 1000 / area if area else None,
                    screen_gradient_norm_sum=float(grad[sel].sum()) if grad is not None else None,
                    screen_gradient_per_reference_pixel=float(grad[sel].sum() / area) if grad is not None and area else None,
                    mean_point_opacity=float(z['opacity'].reshape(-1)[sel].mean()) if kind == 'gradient' and count else None,
                    evidence=str(path.relative_to(ROOT))))
    write_csv(out / 'records.csv', rows)
    events = [json.loads(line) for line in (run / 'structure_events.jsonl').read_text().splitlines()]
    appended = sum(e.get('added', 0) for e in events); pruned = sum(e.get('removed', 0) for e in events)
    with np.load((BASE / 'four_dg/data' / label) / source['canonical_npz']) as z:
        initial = len(z['points'])
    terminal = json.loads((run / 'snapshots/33000.json').read_text())['points']
    assert initial + appended - pruned == terminal, (initial, appended, pruned, terminal)
    summary = dict(state='complete', label=label, source='Saved actual sampling trace, every400-step screen gradients, and all recorded append/prune calls',
        frames=len(points), train_views=data['train_views'], training_renders=len(trace), actual_updates=sum(r['updated'] for r in trace),
        gradient_snapshots=sum(p.name.startswith('gradient') for p in files), structure_record_files=sum(p.name.startswith('structure') for p in files),
        structure_balance=dict(initial=initial, appended=appended, pruned=pruned, terminal=terminal, passed=True),
        core_function_sha256=sha(ROOT / 'scripts/prepare_diagnostics.py'), exposure=[],
        limitations=['Projected center assignment is view-dependent and may be occluded; not semantic Gaussian ownership',
          'Gradient snapshots are sparse, not the full trajectory; structure part assignment uses the camera sampled at that event',
          'Gradient/point normalization uses reference core area; unknown is preserved',
          'This post-hoc train-only diagnostic is not validated donor selection or evidence transfer',
          'Historical32 have sampling traces/counts; comparable per-step gradients/densification history not_recorded'])
    for part in PARTS:
        partrows = [r for r in exposure if r['part'] == part and r['reference_core_pixels']]
        pixels = sum(r['reference_core_pixels'] for r in partrows)
        summary['exposure'].append(dict(part=part, valid_view_time_pairs=len(partrows), expected_view_time_pairs=len(counts), reference_pixels=pixels,
            pixel_renders=sum(r['reference_core_pixel_renders'] for r in partrows),
            mean_render_exposure_per_reference_pixel=sum(r['reference_core_pixel_renders'] for r in partrows) / pixels if pixels else None,
            min_pair_renders=min(r['training_renders'] for r in partrows) if partrows else None,
            max_pair_renders=max(r['training_renders'] for r in partrows) if partrows else None))
    canvas = Image.new('RGB', (1056, 2010), '#282828'); draw = ImageDraw.Draw(canvas)
    for row, view in enumerate([1, 9, 17]):
        for col, frame in enumerate([0, (len(points) - 1) // 2, len(points) - 1]):
            roi, _ = core(view, frame)
            rgba = np.asarray(Image.open(sample / 'data/images' / f'cam{view:02d}' / f'{frame:06d}.png'), np.float32) / 255
            rgb = (rgba[..., :3] * rgba[..., 3:] + .5 * (1 - rgba[..., 3:])) * 255
            for p, color in enumerate(COLORS, 1): rgb[roi == p] = .55 * rgb[roi == p] + .45 * color
            canvas.paste(Image.fromarray(np.rint(rgb).astype(np.uint8)), (352 * col, 670 * row + 30))
            draw.text((352 * col + 5, 670 * row + 8), f'{label} TRAIN v{view:02d} f{frame} CORE/UNKNOWN', fill='white')
    canvas.save(out / 'review.png')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(label=label, state='complete', structure_balance=summary['structure_balance'])), flush=True)


if __name__ == '__main__': main()
