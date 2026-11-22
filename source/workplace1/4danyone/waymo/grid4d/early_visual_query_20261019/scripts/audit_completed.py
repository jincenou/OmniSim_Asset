"""Read-only audit of immutable completed training runs, independent of markers.

This never launches optimization and never turns a missing result into a pass.
It checks actual recovery tensors, optimizer counters, paired exposure records,
and file hashes. Full experiment/visual completion is a separate decision.
"""
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

NEW = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix('.writing')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def counters(counts, step, arm):
    assert counts['updates'] == step
    assert counts['target_renders'] == 2 * step
    assert counts['alpha_renders'] == 2 * max(0, step - 15000)
    assert counts['support_times'] == (max(0, step - 2999) if arm in ['R2', 'R3'] else 0)


def main():
    torch.set_num_threads(2)
    plan = read(NEW / 'PLAN.json')
    target = NEW / 'research/completed_training_audit.json'
    prior = read(target) if target.exists() else {'runs': {}}
    results = prior['runs']
    for rel, expected in plan['implementation_sha256'].items():
        assert sha(NEW / rel) == expected, rel
    for rel, expected in plan['evaluation']['expanded_ROI_sha256'].items():
        assert sha(NEW / rel) == expected, rel
    for label in plan['labels']:
        inp = plan['inputs'][label]
        assert sha(inp['manifest']) == inp['manifest_sha256']
        assert sha(inp['initialization']) == inp['initialization_sha256']
        assert sha(inp['anchor_file']) == inp['anchor_sha256']
        assert sha(NEW / 'data' / label / 'schedule.json') == inp['schedule_sha256']
        schedule = read(NEW / 'data' / label / 'schedule.json')
        assert len(schedule) == 32999
        cameras = read(NEW / 'data' / label / 'cameras.json')
        for frame, v1, v2 in schedule:
            assert 0 <= frame < inp['frame_count'] and v1 != v2
            assert v1 in cameras['train_views'] and v2 in cameras['train_views']
            assert v1 not in [0, 8, 16] and v2 not in [0, 8, 16]
        for arm in ['warmup'] + plan['arms']:
            key = label + '_' + arm
            run = NEW / 'runs' / key
            if not (run / 'training.complete.json').exists():
                continue
            if key in results:
                # Previously audited immutable files must remain present and sized.
                for entry in results[key]['recoveries']:
                    assert Path(entry['path']).stat().st_size == entry['logical_bytes']
                continue
            finish = 2999 if arm == 'warmup' else 32999
            done = read(run / 'training.complete.json')
            assert done['step'] == finish
            counters(done['counts'], finish, arm)
            trace = [json.loads(line) for line in (run / 'trace.jsonl').read_text().splitlines()]
            steps = [r['step'] for r in trace]
            assert steps == sorted(set(steps)) and steps[-1] == finish
            for row in trace:
                assert [row['frame'], *row['views']] == schedule[row['step'] - 1]
            records = []
            query_start = None
            parameter_changes = {}
            optimizer_steps = None
            for step in ([2999] if arm == 'warmup' else [5000, 15000, 18000, 32999]):
                path = run / f'recovery_{step}.pt'
                snap = read(run / f'snapshot_{step}.json')
                h = sha(path)
                assert h == snap['recovery_sha256']
                state = torch.load(path, map_location='cpu')
                assert state['schema'] == 'evq_recovery_v1' and state['step'] == step
                assert state['extra']['schedule_sha256'] == inp['schedule_sha256']
                counters(state['extra']['counts'], step, arm)
                assert state['extra']['counts'] == snap['counts']
                n = len(state['gaussians']['_xyz'])
                assert n == snap['points'] and all(len(v) == n for v in state['gaussians'].values())
                for tensor in state['gaussians'].values():
                    assert torch.isfinite(tensor).all()
                records.append(dict(step=step, path=str(path), sha256=h,
                                    logical_bytes=path.stat().st_size, points=n,
                                    plan_sha256=state['extra']['plan_sha256']))
                if arm in ['R1', 'R2', 'R3']:
                    required = ['query.weight', 'key.weight', 'value.weight', 'offset.weight', 'head.weight'] if arm == 'R3' else ['head.weight']
                    if step == 5000:
                        query_start = {k: state['query'][k].clone() for k in required}
                    if step == 32999:
                        parameter_changes = {k: float((state['query'][k] - query_start[k]).abs().max()) for k in required}
                        # This is an observation, not a new quality admission threshold.
                        optimizer_steps = sorted(set(int(s['step']) for s in state['query_optimizer']['state'].values()))
                        assert optimizer_steps == [30000], (key, optimizer_steps)
                else:
                    assert state['query'] is None and state['query_optimizer'] is None
                del state
                gc.collect()
            if arm != 'warmup':
                resume = read(run / 'resume.json')
                shared = NEW / 'runs' / (label + '_warmup') / 'recovery_2999.pt'
                assert Path(resume['source']) == shared and resume['actual_step'] == 2999
                assert resume['sha256'] == read(shared.parent / 'snapshot_2999.json')['recovery_sha256']
            diagnostics = [r for r in trace if r['diagnostics']]
            ranges = {k: dict(min=min(r['diagnostics'][k] for r in diagnostics),
                              max=max(r['diagnostics'][k] for r in diagnostics),
                              last=diagnostics[-1]['diagnostics'][k])
                      for k in diagnostics[-1]['diagnostics']} if diagnostics else {}
            gradient = {k: max(r['query_grad_abs_sum'].get(k, 0.) for r in trace)
                        for k in parameter_changes}
            results[key] = dict(passed=True, scope='training_recovery_and_exposure_only',
                                terminal_updates=finish, recoveries=records,
                                recorded_observations_match_frozen_schedule=True,
                                trace_rows=len(trace), diagnostic_ranges=ranges,
                                gradient_max=gradient,
                                query_parameter_max_change_5000_to_32999=parameter_changes,
                                query_optimizer_steps=optimizer_steps,
                                gaussian_update_counts=done['counts']['gaussian_updates_by_group'])
            save(target, dict(at=datetime.now(timezone.utc).isoformat(), runs=results,
                              goal_complete=False, current_plan_sha256=sha(NEW / 'PLAN.json'),
                              unobserved_steps_not_claimed_individually_logged=True))
            print('audited', key, finish, flush=True)
    pending = [l + '_' + a for l in plan['labels'] for a in plan['arms'] if l + '_' + a not in results]
    save(target, dict(at=datetime.now(timezone.utc).isoformat(), runs=results,
                      training_audit_complete=not pending, pending=pending,
                      goal_complete=False, current_plan_sha256=sha(NEW / 'PLAN.json'),
                      unobserved_steps_not_claimed_individually_logged=True,
                      finite_gaussians_verified=True,
                      visual_quality_and_full_deliverables_require_separate_audit=True))
    print('pending', pending, flush=True)


if __name__ == '__main__':
    main()
