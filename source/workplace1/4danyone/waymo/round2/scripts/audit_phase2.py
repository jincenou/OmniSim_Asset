"""Verify the bounded phase-two delivery against immutable local evidence."""
from datetime import datetime, timezone
from pathlib import Path
import csv
import json

import imageio_ffmpeg

from audit_current_delivery import main as audit_assets, read, sha, R
from backend_variants import RUNS, completed_variants


def main():
    audit_assets()
    delivered = read(R / 'delivery_requirement_audit.json')
    assert delivered['technical_assets'] == delivered['distinct_tracks'] == 18
    with (R.parents[1] / 'research/gpus.csv').open() as handle:
        devices_by_id = {int(row['index']): row['uuid'].removeprefix('GPU-')
            for row in csv.DictReader(handle, skipinitialspace=True)}
    alternatives = []
    for label in RUNS:
        for item in completed_variants(label):
            root, asset, metrics = item['root'], item['asset'], item['metrics']
            proof = read(root / 'demo/validation.json')
            train = read(root / 'complete.json')
            assert train['coarse_iterations'] == 3000 and train['fine_iterations'] == 30000
            if 'total_updates' in metrics:
                assert metrics['total_updates'] == 33000
            manifest = Path(asset['source_manifest'])
            assert sha(manifest) == asset['source_manifest_sha256']
            data = read(manifest)
            assert data['frame_count'] == asset['frames']
            if 'frame_count' in metrics:
                assert metrics['frame_count'] == asset['frames']
            else:
                assert item['run'] in ['walk063_recipe30k_v1', '0084_recipe30k_v1']
                assert asset['frames'] == 121
                assert max(s['frame'] for s in metrics['samples']) == 120
            assert data['time_denominator'] == asset['frames'] - 1
            assert data['heldout_views'] == [0, 8, 16]
            assert len(data['train_views']) == 21
            assert not set(data['train_views']).intersection(data['heldout_views'])
            for key in ('canonical_ply', 'deformation_weights', 'deformation_table', 'deformation_accum'):
                path = root / asset[key]
                assert sha(path) == asset['asset_sha256'][path.name] == metrics['asset_files'][path.name]
            assert proof['asset_files'] == metrics['asset_files']
            assert sha(root / 'evaluation/results.json') == proof['evaluation_sha256']
            assert sha(Path(proof['job'])) == proof['job_sha256']
            for directory, videos in [('evaluation', metrics['videos']), ('demo', proof['videos'])]:
                for name, record in videos.items():
                    assert sha(root / directory / name) == record['sha256']
                    assert record['frames'] == (72 if name == 'orbit_fixed_time.mp4' else asset['frames'])
            assert len(proof['details']) == 9 and proof['production_approved'] is False
            assert metrics['new_process_replay_max_abs'] == 0
            assert len(metrics['samples']) == 117
            assert {s['view'] for s in metrics['samples']} == {0, 8, 16}
            sampling = read(root / 'point_cloud/iteration_30000/sampling.json')
            for stage, budget in [('coarse', 3000), ('fine', 30000)]:
                counts = sampling[stage]
                assert len(counts) == 24 and all(len(row) == asset['frames'] for row in counts)
                assert sum(map(sum, counts)) == budget
                assert all(sum(counts[v]) == 0 for v in [0, 8, 16])
            devices = [train['device'], metrics['device'], proof['device']]
            assert all(d['physical_gpu'] in [4, 5, 6, 7] and d['cuda_uuid'] for d in devices)
            assert all(d['cuda_uuid'].removeprefix('GPU-') == devices_by_id[d['physical_gpu']]
                for d in devices)
            alternatives.append({'run': item['run'], 'track_label': label, 'frames': asset['frames'],
                'gaussians': asset['gaussians'], 'updates': 33000,
                'asset_hashes_match': True, 'video_hashes_match_full_decode_records': True,
                'heldout_optimizer_updates': 0, 'replay_max_abs': 0,
                'devices': devices, 'production_approved': False})
    assert len(alternatives) == 14

    root_demo = R / 'reconstruction/outputs/walk063_rootlocal24/demo'
    replay = read(root_demo / 'replay_validation.json')
    replay_videos = {}
    for name, expected in replay['videos'].items():
        path = root_demo / name
        reader = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
        metadata = next(reader)
        count = sum(1 for _ in reader)
        assert count == expected['frames']
        assert tuple(metadata['size']) == tuple(expected['size'])
        assert abs(metadata['fps'] - expected['fps']) < 1e-6
        replay_videos[name] = {'frames': count, 'fps': metadata['fps'],
            'size': list(metadata['size']), 'sha256': sha(path)}
    assert max(c['joint_translation_render_max_abs'] for c in replay['translation_checks']) == 0
    assert replay['coordinate_recomposition_max_abs'] < 1e-12
    root_metrics = read(root_demo.parent / 'quality_metrics/metrics.json')
    assert sha(Path(root_metrics['checkpoint'])) == root_metrics['checkpoint_sha256']
    assert root_metrics['ply_roundtrip']['render_max_abs'] == 0
    assert max(root_metrics['ply_roundtrip']['all_parameter_max_abs'].values()) == 0
    assert len([s for s in root_metrics['samples'] if s['split'] == 'validation_generated']) == 117
    geometry = read(R / 'root_local/pilot_063/validation.json')
    assert geometry['checks']['root_relative_joint_vectors_max_abs'] < 1e-6
    assert geometry['checks']['vertices_world_vertical_max_abs'] == 0
    assert len(read(root_demo / 'details/validation.json')['records']) == 15
    gradient = read(R / 'cuda_alpha_probe.json')
    assert gradient['state'] == 'actual_cuda_rasterizer_gradient_passed'

    for check in read(R / 'quality_results.json')['checks']:
        assert all(check[key] for key in ['same_initial_parameters_and_normalization',
            'same_config_except_train_camera_ids_and_output', 'same_evaluated_RGBA_bytes_and_crops',
            'all_train_camera_frame_pairs_seen', 'no_heldout_or_excluded_camera_updates'])
    for check in read(R / 'native_results.json')['checks']:
        assert check['native_targets_paths_hashes_and_crops_equal']
        assert check['initial_parameter_hashes_and_PCA_equal']
    for name in ['budget_results.json', 'native_results.json', 'optimization_results.json']:
        for row in read(R / name)['rows']:
            if 'metrics' in row and isinstance(row['metrics'], str):
                assert Path(row['metrics']).is_file()

    evidence = {
        '1_existing_jobs': ['status.json', 'progress_live_20261014_handoff.json'],
        '2_runtime_and_gpu': ['gpu_sharing_policy.json', 'cuda_alpha_probe.json',
            'alternatives/4dgaussians/reports/runtime_preflight.json'],
        '3_provenance': ['sample_manifest.json', 'calibration_audit.json', 'native_preview_manifest.json',
            'pipeline.json', 'code_manifest.json'],
        '4_varied_demos': ['gallery_manifest.json', 'reviews/baseline12.json', 'action_review.json'],
        '5_view_controls': ['quality_results.json', 'quality_trainer_preflight.json'],
        '6_quality_controls': ['quality_audit.json', 'cuda_alpha_probe.json', 'budget_results.json',
            'native_results.json', 'optimization_results.json', 'validation_comparison.json',
            'alternatives/4dgaussians/reports/temporal_recipe_results.json',
            'alternatives/4dgaussians/reports/native_validation_results.json'],
        '7_walking_root_local': ['root_local/pilot_063/validation.json',
            'reconstruction/outputs/walk063_rootlocal24/demo/replay_validation.json',
            'reviews/walking_replay.json', 'alternatives/4dgaussians/reports/geometry_review.json'],
        '8_metrics_and_delivery': ['delivery_requirement_audit.json', 'benchmark/protocol.json',
            'benchmark/results.json', 'research/latest_metrics.md', 'gallery/index.html'],
        '9_scope_and_handoff': ['REPORT.md', 'HANDOFF.md', 'CONTINUE_PROMPT.txt', 'experiment_matrix.json']}
    required_files = []
    for section, names in evidence.items():
        for name in names:
            path = R / name
            assert path.is_file(), (section, name)
            required_files.append({'section': section, 'path': name, 'sha256': sha(path)})
    result = {'updated_at': datetime.now(timezone.utc).isoformat(),
        'state': 'phase2_technical_and_diagnostic_deliverables_verified_quality_limits_retained',
        'scope': 'Complete the bounded phase-two demos, controlled diagnostics, walking replay and handoff. '
            'The objective explicitly does not require12 visually approved assets;24/100 expansion is conditional.',
        'technical_assets': 18, 'distinct_tracks': 18, 'new_backend_variants': alternatives,
        'objective_reference': {'path': 'research/goal_reference_20261014.txt',
            'sha256': sha(R / 'research/goal_reference_20261014.txt')},
        'generation_quality_formally_approved_count': None,
        'generation_quality_scope': 'Sampled screens permit exploratory reconstruction; '
            'no uniform production-grade generation approval count was established.',
        'production_approved': 0, 'replay_video_full_decode': replay_videos,
        'root_transform': replay['transform_rule'], 'covariance_rotation_SH': replay['covariance_rotation_SH'],
        'strict_view_controls': delivered['strict_view_controls'], 'requirements': required_files,
        'evaluation_scope': 'Heldout generated observations used for validation, not real novel-view test GT. '
            'Video full decoding is technical validation; perceptual checks remain sampled, not production approval.',
        'authorized_cleanup': 'Intermediate checkpoints removed after retaining terminal assets and their metrics/records; '
            'see ../../cleanup/receipt.json. Deleted snapshot paths are historical evidence, not runnable assets.',
        'future_quality_and_integration': ['Remaining local foot/hand failures despite better aggregate metrics',
            'Reliable source action and moving-camera translation separation',
            'Waymo metric scale, world trajectory, ground contact and scene occlusion',
            'Arbitrary SMPL-X action retargeting; current models replay their fitted clip'],
        'expansion': 'Do not automatically restart100;walking grid61 is a candidate,standing retainsgrid25reference.'}
    (R / 'phase2_completion_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'state': result['state'], 'tracks': 18, 'variants': len(alternatives),
        'root_replay_videos': len(replay_videos), 'production_approved': 0}), flush=True)


if __name__ == '__main__':
    main()
