"""Close the eight-identity experiment only after completed evidence and scoped review."""
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

NEW = Path(__file__).resolve().parents[1]
V3 = NEW.parent / 'optimization_quality_v3_20261018'


def read(path):
    def invalid(value):
        raise ValueError(value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    plan = read(NEW / 'PLAN.json')
    state = read(NEW / 'STATUS.json')
    assert state['state'] == 'training_evaluation_complete_review_pending'
    assert len(state['jobs']) == 10
    assert all(j['state'] == 'complete' and all(s['exit_code'] == 0 for s in j['stages']) for j in state['jobs'].values())
    analysis = read(NEW / 'ANALYSIS_STATUS.json')
    assert analysis['state'] == 'numeric_and_asset_audit_complete_visual_report_pending'
    profiles = read(NEW / 'PROFILE_STATUS.json')
    assert profiles['state'] == 'complete' and len(profiles['jobs']) == 18
    assert all(j['exit_code'] == 0 for j in profiles['jobs'].values())
    inputs = read(NEW / 'research/input_audit.json')
    recovery = read(NEW / 'research/recovery_audit.json')
    assert inputs['state'] == recovery['state'] == 'passed'
    assert len(inputs['inputs']) == 8
    assert sum(x['training_files'] + x['heldout_files'] for x in inputs['inputs']) == 14592
    assert len(recovery['exports']) == len(recovery['independent_replays']) == 18
    assert len(recovery['optimizer_increments']) == 8
    previous = read(V3 / 'completion_audit.json')
    assert previous['state'] == 'completed_requested_bounded_round'
    for name, expected in previous['artifact_hashes'].items():
        assert sha(V3 / name) == expected, name
    assets = read(NEW / 'ASSETS.json')['assets']
    assert len(assets) == 16 and sum(a['reused'] for a in assets) == 6
    for asset in assets:
        run = Path(asset['asset'])
        iteration = 53000 if asset['method'] == 'A' else 33000
        snapshot = read(run / 'snapshots' / f'{iteration}.json')
        for name, expected in snapshot['files'].items():
            assert sha(run / name) == expected, (run, name)
    gates = read(NEW / 'gate_results.json')
    assert {g['label'] for g in gates['checks']} == set(plan['labels'])
    assert len(gates['reference_denominator_checks']) == 8
    assert read(NEW / 'cohort_metrics.json')['state'] == 'all_terminal_pairs_available'
    costs = list(csv.DictReader((NEW / 'costs.csv').open()))
    assert sum(r['category'] == 'training' for r in costs) == 18
    assert sum(r['category'] == 'full_sequence_asset_only_replay' for r in costs) == 18
    visual = read(NEW / 'visual_review.json')
    assert visual['state'] == 'sampled_eight_identity_review_complete_full_video_unknown'
    assert set(visual['cases']) == set(plan['labels'])
    for label, case in visual['cases'].items():
        folder = Path(case['folder'])
        for name, expected in case['sha256'].items():
            assert sha(folder / name) == expected, (label, name)
    media = []
    for label in plan['labels']:
        folder = V3 / 'comparison' / f'{label}_A_fusion' if label in ['N01', 'N07'] else NEW / 'comparison' / label
        validation = read(folder / 'validation.json')
        assert len(validation['videos']) == 3
        assert sha(validation['selection_path']) == validation['selection_sha256']
        frames = 121 if label in ['N07', 'N11'] else 61
        for video in validation['videos']:
            assert video['frames'] == frames and video['height'] == 672 and video['fps'] == 25
            assert video['width'] == (2112 if label in ['N01', 'N07'] else 1408)
            assert sha(folder / video['path']) == video['sha256']
        media.append(dict(label=label,folder=str(folder),validation=str(folder / 'validation.json'),videos=validation['videos'],decoded_by='unchanged publisher full-frame decode',human_full_motion_review=False))
    save(NEW / 'comparison/media_validation.json', dict(state='passed_24_complete_videos', families=media))
    page = NEW / 'comparison/index.html'
    broken = []
    for ref in re.findall(r'(?:href|src)="([^"]+)"', page.read_text()):
        if not ref.startswith(('http:', 'https:', '#')) and not (page.parent / ref).exists():
            broken.append(ref)
    assert not broken, broken
    assert '最终交付' in (NEW / 'REPORT.md').read_text()
    gates.update(state='fixed_numeric_gates_and_scoped_visual_review_complete', visual_review='visual_review.json', uniform_new_recipe_accepted=False)
    for item in gates['checks']:
        item['visual'] = visual['cases'][item['label']]['visual_state']
        item['decision'] = 'rejected_numeric' if item['numeric_state'] == 'fail' else 'numeric_only_no_full_visual_acceptance'
    save(NEW / 'gate_results.json', gates)
    state.update(state='completed_eight_identity_experiment', completed_at=datetime.now(timezone.utc).isoformat(), counts=dict(identities=8, terminal_assets=16, reused_terminal_assets=6, new_training_branches=10, new_stage_exports=18, complete_comparison_videos=24), quality='Full A research baseline retained; no universal fused replacement accepted', background_work_remaining=False, completion_audit='completion_audit.json')
    save(NEW / 'STATUS.json', state)
    requirements = [
        ('Eight existing legal identities, seven main plus one reserve; failures not replaced', ['PLAN.json', 'cohort_metrics.json']),
        ('Frozen inputs, configs, training source, prior dependencies and six reused terminals', ['research/input_audit.json', 'research/recovery_audit.json']),
        ('Ten new branches, eighteen complete stage packages and optimizer recovery checks', ['STATUS.json', 'research/recovery_audit.json']),
        ('Sixteen final assets with matching snapshot/runtime contract', ['ASSETS.json', 'research/recovery_audit.json']),
        ('All fixed global/part/temporal comparisons; unknown values preserved', ['paired_metrics.csv', 'stage_metrics.csv', 'part_metrics.csv', 'gate_results.json']),
        ('Twenty-four complete comparison videos; explicit scoped visual inspection', ['comparison/media_validation.json', 'visual_review.json']),
        ('Measured training/evaluation/full-sequence replay costs; shared-load limitations', ['costs.csv', 'cost_summary.json', 'PROFILE_STATUS.json']),
        ('Final report, metrics and working gallery links', ['REPORT.md', 'METRICS.md', 'comparison/index.html']),
    ]
    artifact_names = ['PLAN.json', 'FUSION_PLAN.json', 'STATUS.json', 'ANALYSIS_STATUS.json', 'PROFILE_STATUS.json', 'REPORT.md', 'METRICS.md', 'paired_metrics.csv', 'stage_metrics.csv', 'cohort_metrics.json', 'part_metrics.csv', 'gate_results.json', 'ASSETS.json', 'costs.csv', 'cost_summary.json', 'visual_review.json', 'comparison/index.html', 'comparison/metrics.png', 'comparison/media_validation.json', 'research/input_audit.json', 'research/recovery_audit.json']
    save(NEW / 'completion_audit.json', dict(state='completed_requested_eight_identity_expansion', at=datetime.now(timezone.utc).isoformat(), counts=state['counts'], requirements=[dict(requirement=name,state='passed_execution_or_documented_outcome',evidence=paths) for name,paths in requirements], quality_success_required_for_experiment_completion=False, limitations=['Generated H is development reference, not real Waymo ground truth or independent test.', 'Sampled still inspection and full decoding do not establish full-video motion quality.', 'N02 leg cores remain unknown; shared-load wall times are not controlled algorithm speedups.', 'One seed and independent CUDA trajectories do not establish statistical significance.', 'Dynamic asset requires the matched runtime; no arbitrary motion driving or metric world scale claim.'], artifact_hashes={n:sha(NEW/n) for n in artifact_names}, final_analysis_sources={str(p.relative_to(NEW)):sha(p) for p in sorted((NEW/'scripts').glob('*.py'))}))
    print('Eight-identity experiment delivery audit passed')


if __name__ == '__main__':
    main()
