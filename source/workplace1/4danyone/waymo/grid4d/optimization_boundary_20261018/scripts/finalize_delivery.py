"""Verify concrete delivery artifacts before declaring this bounded round complete."""
import ast
import csv
import hashlib
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
import cv2

NEW = Path(__file__).resolve().parents[1]
ROOT = NEW.parent
FIELDS = ['full_psnr','foreground_psnr','ssim','crop_lpips','silhouette_iou','outside_alpha_mean']

def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        while chunk := file.read(8*1024*1024):
            h.update(chunk)
    return h.hexdigest()

def record(path):
    path = Path(path)
    assert path.is_file()
    return dict(path=str(path.relative_to(NEW)), bytes=path.stat().st_size, sha256=sha(path))

def main():
    now = datetime.now(timezone.utc).isoformat()
    plan = read(NEW/'PLAN.json')
    plan_sha = sha(NEW/'PLAN.json')
    assert plan_sha == (NEW/'plan_hash.txt').read_text().strip()
    assert sha(NEW/'REQUEST.txt') == plan['request_sha256']
    dependencies = read(NEW/'research/dependencies.json')
    for item in dependencies.values():
        assert sha(item['path']) == item['sha256'], item['path']
    for path, digest in read(NEW/'research/training_implementation.json').items():
        assert sha(path) == digest, path
    assets = []
    pair_checks = []
    validation = read(NEW/'comparison/validation.json')
    gates = read(NEW/'gate_results.json')
    assert gates['repeat_authorized'] is False and gates['repeat_state'] == 'not_triggered_numeric_gates_failed'
    assert all(row['state']=='fail' for row in gates['comparisons'])
    recovery = read(NEW/'research/recovery_audit.json')
    assert recovery['state'] == 'passed' and len(recovery['snapshots']) == 18
    table = list(csv.DictReader((NEW/'paired_metrics.csv').open()))
    assert len(table) == 18
    for label in ['N04','N11']:
        item = plan['identities'][label]
        for previous in item['files'].values():
            assert sha(previous['path']) == previous['sha256']
        for name in ['training','evaluation']:
            assert sha(item['data'][name]) == item['data'][name+'_sha256']
        train = read(item['data']['training'])
        assert len(train) == 21*item['frames'] and all(r['view'] not in [0,8,16] for r in train)
        schedule = read(item['sampling']['primary']['path'])
        lrs = read(item['lr']['path'])
        assert sha(item['sampling']['primary']['path']) == item['sampling']['primary']['sha256']
        assert sha(item['lr']['path']) == item['lr']['sha256']
        assert len(schedule) == len(lrs) == 5000
        band = read(NEW/'data'/label/'bands/complete.json')
        assert band['heldout_content_reads'] == 0 and band['empty_band_count'] == 0
        for path, digest in band['files'].items():
            assert sha(path) == digest
        old = read(Path(item['start_run']).parents[1]/'results'/f'{label}_alpha'/'metrics.json')
        old_keys = [(r['view'],r['frame'],r['crop_xyxy'],r['input_sha256']) for r in old['samples']]
        parts_reference = None
        for arm in ['A','U','B']:
            run = NEW/'runs/primary'/f'{label}_{arm}'
            done = read(run/'complete.json')
            assert done['stage_updates']==5000 and done['total_actual_updates']==52999 and done['points']==item['points']
            assert done['costs']['training_render_calls']==10000 and done['costs']['training_backward_calls']==5000
            traces = list(run.glob('trace_*.jsonl'))
            assert len(traces)==1
            trace = [json.loads(line) for line in traces[0].read_text().splitlines()]
            assert len(trace)==5000 and sha(traces[0])==done['trace_files'][traces[0].name]
            for i, row in enumerate(trace):
                assert row['stage_update']==i+1 and row['global_actual_updates']==48000+i
                assert row['uid']==schedule[i]['uid'] and row['view']==schedule[i]['view'] and row['frame']==schedule[i]['frame']
                assert row['points']==item['points']
                assert all(value==lrs[i][name] for name,value in row['lr'].items())
                assert math.isclose(row['loss'],row['rgb_loss']+plan['lambda_alpha']*row['old_alpha_l1']+row['regularizer']+plan['delta']*row['added_alpha_l1'],rel_tol=3e-6,abs_tol=1e-7)
            metrics = read(NEW/'results/primary'/run.name/'metrics.json')
            assert [(r['view'],r['frame'],r['crop_xyxy'],r['input_sha256']) for r in metrics['samples']] == old_keys
            assert len(metrics['samples']) == 39 and metrics['actual_optimizer_updates']==52999
            displayed = next(r for r in table if r['label']==label and r['method']==arm)
            for field in FIELDS:
                assert float(displayed[field])==metrics['mean'][field]
                assert math.isclose(sum(s[field] for s in metrics['samples'])/39,metrics['mean'][field],rel_tol=1e-12,abs_tol=1e-12)
            parts = list(csv.DictReader((NEW/'results/primary'/run.name/'parts.csv').open()))
            denominator = [(r['view'],r['frame'],r['part'],r['reference_core_pixels'],r['temporal_valid_pixels']) for r in parts]
            assert len(parts) == 18*item['frames']
            if parts_reference is None:
                parts_reference = denominator
            assert parts_reference == denominator
            snapshots = []
            for iteration, extra in [(49000,1000),(51000,3000),(53000,5000)]:
                snapshot = read(run/'snapshots'/f'{iteration}.json')
                replay = read(run/'validation'/f'{iteration}.json')
                assert snapshot['actual_optimizer_updates']==47999+extra and snapshot['points']==item['points']
                assert replay['state']=='passed' and replay['source_data_reads']==0 and replay['reference_count']==12
                assert replay['independent_replay_max_abs_error'] < 1e-5 and replay['background_composition_max_abs_error'] < 1e-5
                assert any(t['frame']==12.25 for t in replay['time_checks'])
                files = []
                for path, digest in snapshot['files'].items():
                    entry = record(run/path)
                    assert entry['sha256']==digest
                    files.append(entry)
                snapshots.append(dict(nominal_iteration=iteration,actual_updates=47999+extra,files=files,reference=record(run/'snapshots'/f'{iteration}_reference.npz'),replay=record(run/'validation'/f'{iteration}.json')))
            assert (run/'recovery.pt').resolve()==(run/'recoveries/53000.pt').resolve()
            assets.append(dict(identity=label,arm=arm,root=str(run.relative_to(NEW)),final_actual_updates=52999,points=item['points'],dynamic_components=[record(run/n) for n in ['config.json','asset_camera_data.json']],snapshots=snapshots,recovery_link='recovery.pt -> recoveries/53000.pt',metrics=record(NEW/'results/primary'/run.name/'metrics.json'),quality_gate='control_only' if arm=='A' else 'failed_predeclared_gate'))
        probe = read(NEW/'results/analysis'/label/'query_readiness.json')
        assert len(probe['anchors'])==6 and len(probe['candidates'])==48 and probe['optimizer_steps']==0 and probe['query_training_executed'] is False
        assert all(c['target_view'] not in [0,8,16] and c['donor']['view'] not in [0,8,16] for c in probe['candidates'])
        cutoff = read(NEW/'research'/f'{label}_cutoff_audit.json')
        assert cutoff['optimizer_updates']==0 and cutoff['pixels_with_changed_contributors']>0
        pair_checks.append(dict(label=label,all_5000_samples_and_lrs_match_frozen_plan=True,all_loss_components_match=True,reference_crops_hashes_and_part_denominators_preserved=True,query_candidates_train_only=True))
    videos = []
    for video in validation['videos']:
        path = NEW/'comparison'/video['path']
        assert sha(path)==video['sha256']
        cap = cv2.VideoCapture(str(path)); count=0
        while True:
            ok, frame=cap.read()
            if not ok: break
            assert frame.shape==(672,2112,3)
            count+=1
        cap.release()
        assert count==video['frames']
        videos.append(dict(path=str(path.relative_to(NEW)),frames=count,sha256=video['sha256']))
    html = NEW/'comparison/index.html'
    local_links=[]
    for value in re.findall(r'(?:src|href)="([^"]+)"',html.read_text()):
        assert (html.parent/value).is_file(), value
        local_links.append(value)
    for script in (NEW/'scripts').glob('*.py'): ast.parse(script.read_text())
    for path in NEW.rglob('*.json'): read(path)
    save(NEW/'ASSETS.json',dict(schema='grid4d_boundary_assets_v1',at=now,assets=assets,runtime=str(ROOT/'source'),environment='/mnt/ojc/miniconda3/envs/grid4d',playback_script=str(ROOT/'scripts/replay_asset.py'),independent_validation_script='scripts/validate_boundary.py',limitations='PLY alone is not the dynamic asset. Requires matched deform/hash, config, cameras and runtime; known-time replay, not new-action control or real-world registration.'))
    requirements = [
        (1,'scope_and_frozen_backend','passed',['PLAN.json','research/runtime_provenance.json']),
        (2,'original_evidence_and_full_recovery','passed',['research/dependencies.json','research/recovery_audit.json']),
        (3,'plan_before_training_and_actual_paired_samples','passed',['PLAN.json','plan_hash.txt','comparison/validation.json']),
        (4,'fixed_regions_and_unmasked_rgb_alpha_diagnosis','completed_with_reference_uncertainty',['boundary_diagnosis.csv','local_crop_metrics.csv','visual_review.json']),
        (5,'A_U_B_losses_no_structure_changes','passed',['scripts/train_boundary.py','gradient_checks.json']),
        (6,'full_restore_lr_and_gradient_scope','passed_with_retained_nonsmooth_failure',['research/recovery_audit.json','gradient_checks.json','research/N04_cutoff_audit.json','research/N11_cutoff_audit.json']),
        (7,'readonly_appearance_time_probe','completed_conditional_evidence',['appearance_time_probe.csv','ANCHOR_AUDIT.json']),
        (8,'offline_donor_transfer_and_query_readiness','completed_no_query_training',['query_readiness.json','visual_review.json']),
        (9,'fixed_terminal_evaluation_and_quality_gate','completed_candidates_failed',['paired_metrics.csv','gate_results.json']),
        (10,'conditional_repeat','not_triggered_numeric_gates_failed',['gate_results.json','NEXT_VALIDATION_PLAN.md']),
        (11,'background_execution_replay_and_full_comparisons','passed_with_visual_review_limits',['STATUS.json','EVALUATION_STATUS.json','comparison/validation.json','ASSETS.json']),
        (12,'deliverables_and_five_research_answers','passed',['README_run.md','FINDINGS.md','NEXT_VALIDATION_PLAN.md'])]
    for _,_,_,paths in requirements:
        for path in paths: assert (NEW/path).is_file(), path
    required=['PLAN.json','plan_hash.txt','ANCHOR_AUDIT.json','STATUS.json','EVALUATION_STATUS.json','boundary_diagnosis.csv','gradient_checks.json','appearance_time_probe.csv','query_readiness.json','paired_metrics.csv','gate_results.json','costs.csv','comparison/index.html','ASSETS.json','README_run.md','FINDINGS.md','NEXT_VALIDATION_PLAN.md']
    for name in required: assert (NEW/name).is_file()
    limitations=['Generated development references, not real Waymo novel-view truth; N11 remains separate.','No new candidate passed all numeric protection gates; no automatic repeat, new identity or Query training.','Stable-piece boundary checks pass; original full-image and full-band finite-difference failures remain.','Chronological sample/crop visual review plus full-frame decoding is not exhaustive human frame-by-frame acceptance.','Canonical expected-point donor matches are conditional and can mix surfaces; no established learned Query advantage.','CPU publication, part-evaluation and total human/audit overhead were not individually timed; do not infer total GPU busy time from cost sums.']
    save(NEW/'research/cost_accounting.json',dict(training_counts='Each arm 5000 updates, 10000 renders, 5000 backward calls; one attempt each.',snapshot_io_scope='Per-arm snapshot_and_recovery_seconds includes three snapshots, serialization, hashing and 72 reference renders (12 pairs per snapshot). It is not pure disk I/O.',logging_seconds={a['root']:read(NEW/a['root']/'complete.json')['costs']['logging_seconds'] for a in assets},other_costs='costs.csv separates preparation, diagnostics, evaluation and independent replay.',unmeasured=['individual CPU part-metric time','video publication I/O wall time','total setup and human review time'],gpu_busy_seconds_measured=False))
    for filename,state in [('STATUS.json','completed_no_qualified_candidate'),('EVALUATION_STATUS.json','completed_metrics_comparison_and_gates')]:
        data=read(NEW/filename)
        data.setdefault('queue_terminal_state',data['state'])
        data.update(state=state,delivery_at=now,completion_audit='completion_audit.json',quality_gate='failed_candidates',repeat_started=False)
        save(NEW/filename,data)
    validation.update(visual_quality_review='sampled_review_completed_exhaustive_acceptance_not_established',visual_review_record='../visual_review.json')
    save(NEW/'comparison/validation.json',validation)
    audit=dict(state='completed_requested_bounded_round',at=now,plan_sha256=plan_sha,quality_outcome='no_uniform_candidate_passed',primary_training_runs=6,extra_actual_updates_per_run=5000,total_new_optimizer_updates=30000,snapshots_replayed=18,repeat_training_runs=0,query_training_runs=0,old_dependencies_unchanged=len(dependencies),paired_checks=pair_checks,full_decoded_videos=videos,html_local_links_checked=len(local_links),asset_manifest='ASSETS.json',requirements=[dict(section=n,requirement=name,state=state,evidence=paths) for n,name,state,paths in requirements],limitations=limitations,free_disk_GiB=shutil.disk_usage(NEW).free/2**30,required_files=[record(NEW/name) for name in required])
    save(NEW/'completion_audit.json',audit)
    print('Delivery audited: 6 paired runs, 18 replayed snapshots, 6 fully decoded comparisons; no qualified new candidate.', flush=True)

if __name__ == '__main__': main()
