"""Read-only inventory of frozen Random8 inputs, metrics and sampling records."""
import asyncio
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'random8/r20261015_113332'
LABELS = ['N01', 'N02', 'N03', 'N04', 'N06', 'N07', 'N08', 'N11']
BRANCHES = ['ftgs_base5k', 'ftgs_opt15k', '4dgs_grid25', '4dgs_grid61']
KEYS = ['full_psnr', 'foreground_psnr', 'ssim', 'crop_lpips', 'silhouette_iou', 'outside_alpha_mean']
H = [0, 8, 16]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inspect(label):
    base = BASE / 'samples' / label
    mp = BASE / 'four_dg/data' / label / 'fdanyone.json'
    manifest = read(mp)
    frames = manifest['frame_count']
    assert manifest['heldout_views'] == H and frames in (61, 121)
    assert manifest['train_views'] == [v for v in range(24) if v not in H]
    assert {(r['view'], r['frame']) for r in manifest['input_hashes']} == {
        (v, f) for v in range(24) for f in range(frames)}
    for row in manifest['input_hashes']:
        path = base / 'data/images' / f"cam{row['view']:02d}" / f"{row['frame']:06d}.png"
        assert digest(path) == row['sha256'], str(path)
        with Image.open(path) as image:
            assert image.mode == 'RGBA' and image.size == (352, 640)
    initial = base / 'data/init.npz'
    assert digest(initial) == manifest['source_init_sha256']
    canonical = mp.parent / manifest['canonical_npz']
    with np.load(initial) as z, np.load(canonical) as c:
        at_zero = np.isclose(z['times'], 0)
        transformed = np.column_stack([z['positions'][at_zero], np.ones(at_zero.sum())]) @ np.asarray(manifest['source_to_normalized']).T
        error = float(np.abs(transformed[:, :3] - c['points']).max())
        assert error < 2e-6
        assert np.allclose(z['colors'][at_zero] / 255, c['colors'], atol=1e-6)
        assert np.isfinite(c['points']).all()
    methods, input_signatures = {}, set()
    for branch in BRANCHES:
        ep = base / 'evaluations' / branch / 'results.json'
        result = read(ep)
        assert result['heldout_views'] == H and len(result['samples']) == 39
        assert result['protocol']['background'] == .5
        expected_frames = np.rint(np.linspace(0, frames - 1, 13)).astype(int).tolist()
        assert result['sampled_frames'] == expected_frames
        input_signatures.add((result['protocol']['all_input_pairs_sha256'], json.dumps(result['crops'], sort_keys=True)))
        assert all(abs(np.mean([r[k] for r in result['samples']]) - result['mean'][k]) < 1e-9 for k in KEYS)
        run = base / branch if branch.startswith('ftgs') else BASE / 'four_dg/runs' / f'{label}_grid{branch[-2:]}'
        config = read(run / 'experiment.json')
        complete = read(run / ('training_complete.json' if branch.startswith('ftgs') else 'complete.json'))
        if branch.startswith('ftgs'):
            counts = np.asarray(complete['sampling']['counts_by_camera_and_frame'])
            assert counts.shape == (24, frames) and not counts[H].any()
            assert counts.sum() == complete['actual_optimizer_updates']
            assert config['full_config']['data_dir'] == str(base / 'data')
            assert digest(initial) == config['base_init_sha256']
            assert digest(Path(config['full_config']['init_npz_path'])) == config['init_sha256']
            exposures = int(counts.sum())
        else:
            assert digest(mp) == config['input_manifest_sha256']
            trace = np.load(run / 'sampling_trace.npy')
            assert digest(run / 'sampling_trace.npy') == complete['sampling_trace_sha256']
            assert not np.isin(trace[:, 1], H).any()
            assert complete['actual_optimizer_updates'] == 32999
            exposures = len(trace)
        methods[branch] = dict(metrics=result['mean'], protocol=result['protocol'], result_path=str(ep),
            result_sha256=digest(ep), config_path=str(run / 'experiment.json'),
            config_sha256=digest(run / 'experiment.json'), complete=complete,
            training_renders=exposures, heldout_training_samples=0,
            historical_part_densification='not_recorded', historical_part_gradient='not_recorded',
            semantic_part_metrics='pending_ROI_validation')
    assert len(input_signatures) == 1
    source = read(base / 'source/visualization/manifest.json')
    return dict(label=label, cohort='reserve' if label == 'N11' else 'main', frames=frames,
        manifest_path=str(mp), manifest_sha256=digest(mp), verified_RGBA_files=len(manifest['input_hashes']),
        source_init_sha256=digest(initial), canonical_path=str(canonical), canonical_sha256=digest(canonical),
        canonical_t0_transform_max_error=error, same_evaluation_input_and_crop_across_four_methods=True,
        source_observation_count=source['native_observations'], source_comparison='unregistered_only',
        methods=methods)


async def main():
    semaphore = asyncio.Semaphore(2)
    async def task(label):
        async with semaphore:
            result = await asyncio.to_thread(inspect, label)
            (ROOT / 'audit' / f'{label}_inventory.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps({'label': label, 'state': 'inputs_and_frozen_metrics_verified'}), flush=True)
            return result
    results = await asyncio.gather(*(task(label) for label in LABELS))
    rows = [{'scope': r['cohort'], 'label': r['label'], 'method': branch, **r['methods'][branch]['metrics']}
            for r in results for branch in BRANCHES]
    for branch in BRANCHES:
        rows.append({'scope': 'main7_mean', 'label': ','.join(LABELS[:-1]), 'method': branch,
                     **{k: float(np.mean([r['methods'][branch]['metrics'][k] for r in results[:-1]])) for k in KEYS}})
    with (ROOT / 'audit/baseline_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['scope', 'label', 'method', *KEYS])
        writer.writeheader()
        writer.writerows(rows)
    report = dict(state='input_and_frozen_global_metric_audit_passed', updated_at=datetime.now(timezone.utc).isoformat(),
        cases=LABELS, main=LABELS[:-1], reserve=['N11'], branches=BRANCHES,
        original_results_modified=False, total_RGBA_verified=sum(r['verified_RGBA_files'] for r in results),
        preserved_frontend_failures=['N05', 'N09', 'N10'], complete_stage_part_audit=False,
        initialization_source_review={'script': str(BASE / 'scripts/export_sequence.py'),
            'sha256': digest(BASE / 'scripts/export_sequence.py'),
            'finding': 'Historical exporter decoded all24 RGBA files, but hull votes and color accumulation indexed only21 training views. No heldout content contribution identified in source. New Grid4D training will only open21 training image paths.',
            'canonical': 'HUST/Grid4D use common untrained t0 hull; FTGS uses multiple time hulls. These are full recipe comparisons, not identical initialization across all4 methods.'},
        training_background_note='Historical HUST/CEM train white1 and evaluate gray0.5. Requested Grid4D trains/evaluates gray0.5; report this recipe difference explicitly.',
        pending=['Six-part ROI validation; stage evidence and local metrics', 'Motion-compensated temporal diagnostics',
                 'Grid4D deployment, smoke, formal training and separate evaluation'])
    (ROOT / 'audit/inventory.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
