"""Read stage evidence and live PIDs; publish all eight frozen cases atomically."""
import argparse
import asyncio
import csv
import io
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('prepare', 'segment', 'motion', 'conditioning', 'generate', 'validate_generation',
          'generated_preview', 'masks', 'hull', 'manifest', 'geometry', 'configure',
          'native', 'alpha', 'A', 'package', 'replay', 'comparison')


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def atomic(path, text):
    temporary = path.with_name(path.name + f'.{os.getpid()}.writing')
    temporary.write_text(text)
    temporary.replace(path)


def alive(pid):
    if not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 0)
        # A zombie has a PID but cannot advance a stage.
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except (OSError, ProcessLookupError):
        return False


def collect():
    queue = read(ROOT / 'queue_status.json')
    controller_live = alive(queue.get('pid'))
    people = []
    for choice in read(ROOT / 'SELECTED_TRACKS.json')['tracks']:
        ident = choice['id']
        replacement = read(ROOT / 'ASSET_REPLACEMENTS.json').get('people', {}).get(ident, {})
        active = ROOT / replacement.get('root', '.')
        preview_root = str((active / 'previews' / ident).relative_to(ROOT))
        asset_root = active / 'assets' / ident
        # Preserve preparation failures/deferred camera states when the asset
        # queue contains only reviewed people.
        prior = read(ROOT / 'research/first_asset_queue.json').get('people', {}).get(ident, {})
        prior = read(ROOT / 'research/remaining_preparation_queue.json').get('people', {}).get(ident, prior)
        current = queue.get('people', {}).get(ident, prior)
        complete = {s: read(ROOT / 'logs' / f'{ident}_{s}_complete.json') for s in STAGES}
        processes = {s: read(ROOT / 'logs' / f'{ident}_{s}_process.json') for s in STAGES}
        mask = read(ROOT / 'inputs' / ident / 'research/batch_gate_mask.json')
        motion = read(ROOT / 'inputs' / ident / 'research/motion_gate.json')
        replay = read(active / 'previews' / ident / 'independent_replay.json')
        media = read(active / 'previews' / ident / 'media_validation.json')
        stage = current.get('stage', next((s for s in reversed(STAGES) if complete[s]), 'not_started'))
        proc = processes.get(stage, {})
        child_live = alive(proc.get('pid')) or alive(current.get('pid'))
        state, issue = 'pending', ''
        if choice['camera_class'] in ('moving', 'unknown'):
            state, issue = 'camera_interface_not_applicable_pending', 'Moving/unknown original camera; no static approximation applied'
        elif mask and not mask['passed']:
            state, issue, stage = 'segmentation_gate_failed', 'Old mask validity gate failed; no threshold or identity replacement', 'segment'
        elif replay.get('state') == 'asset_only_replay_passed_quality_unreviewed' and media and (asset_root / 'package_manifest.json').is_file():
            state, stage = 'asset_replayed_quality_unreviewed', 'comparison'
        elif current.get('state') == 'failed':
            state, issue = 'stage_failed', current.get('error', '')
        elif child_live:
            state = 'running' if alive(proc.get('pid')) else 'running_or_waiting_shared_gpu'
        elif proc.get('state') == 'failed':
            state, issue = 'stage_failed', f'{stage} exit {proc.get("exit_code")}; inspect logs'
        elif complete['motion'] and not complete['conditioning']:
            allowed = motion.get('variants', {}).get('static_estimated', {}).get('allow_exploratory_generation', False)
            state = 'pose_review_passed_waiting_asset_queue' if allowed else 'pose_source_review_required'
        elif current and controller_live:
            state = current.get('state', 'queued')
        elif complete['prepare']:
            state = 'prepared_waiting_next_stage'
        # Raw runtime existence never means quality acceptance or complete asset.
        asset_complete = bool(replay.get('state') == 'asset_only_replay_passed_quality_unreviewed' and media and (asset_root / 'package_manifest.json').is_file())
        measured = [v['finished'] - v['launched'] for v in processes.values()
                    if v.get('exit_code') == 0 and 'launched' in v and 'finished' in v]
        views = read(ROOT / 'inputs' / ident / 'research/visual_review.json')
        generated_review = read(active / 'frontend' / ident / 'research/generated_visual_review.json')
        asset_review = read(active / 'previews' / ident / 'asset_visual_review.json')
        known_issues = generated_review.get('observations', []) if generated_review.get('state') == 'sampled_generated_video_has_quality_issues' else []
        quality = 'unreviewed' if asset_complete else 'not_assessed'
        if asset_complete and asset_review.get('observations'):
            quality = 'completed_with_quality_issues' if asset_review.get('has_observed_quality_issues') or asset_review.get('state') == 'completed_with_quality_issues' else 'sampled_review_full_temporal_unknown'
        next_step = STAGES[STAGES.index(stage)+1] if stage in STAGES[:-1] else None
        if state in ('segmentation_gate_failed', 'stage_failed', 'camera_interface_not_applicable_pending'):
            next_step = 'Review recorded failure or interface limitation; no automatic retry'
        elif asset_complete:
            next_step = 'Complete visual quality review' if not asset_review else 'Recorded visual review; inspect remaining unknown temporal ranges'
        people.append(dict(id=ident, source=choice['source_id'], source_track=choice['track_id'],
            action_description=choice['reason'], camera_class=choice['camera_class'],
            source_time_span_s=choice['actual_target_source_end_time_s'] - choice['start_time'],
            playback_duration_s=choice['frame_count'] / choice['fps'],
            median_native_bbox_height_px=choice['median_native_bbox_height_px'],
            state=state, stage=stage, issue=issue, asset_complete=asset_complete,
            quality=quality,
            completed_stages=[s for s in STAGES if complete[s]],
            live_pid=proc.get('pid') if alive(proc.get('pid')) else current.get('pid') if child_live else None,
            physical_gpu=proc.get('physical_gpu') if child_live else None,
            gpu_uuid=proc.get('gpu_uuid') if child_live else None,
            exit_code=None if child_live else proc.get('exit_code', current.get('exit_code')),
            log=str(ROOT / 'logs' / f'{ident}_{stage}.log') if proc else current.get('log'),
            next_step=next_step,
            successful_gpu_stage_wall_s=sum(measured) if measured else None,
            timing_scope='Historical baseline subprocess timings; excludes accessory repair branch, queue wait, CPU stages, failures and ongoing stage',
            asset_variant=replacement.get('variant', 'baseline'), preview_root=preview_root,
            known_frontend_observations=known_issues,
            generated_review_path=str((active / 'frontend' / ident / 'research/generated_visual_review.json').relative_to(ROOT)) if generated_review else None,
            input_review_path=f'inputs/{ident}/research/visual_review.json' if views else None,
            asset_review_path=f'{preview_root}/asset_visual_review.json' if asset_review else None,
            asset_observations=asset_review.get('observations', []),
            asset_path=str(asset_root.relative_to(ROOT)) if asset_complete else None))
    return queue, controller_live, people


def repair_status():
    result = []
    for ident in ('W02', 'W04'):
        root = ROOT/'diagnostics'/f'accessory_{ident}_v1'
        if not root.exists():
            continue
        queue = read(root/'queue_status.json')
        row = queue.get('people', {}).get(ident, {})
        stage = row.get('stage', 'generation')
        process = read(root/'logs'/f'{ident}_{stage}_process.json')
        if not queue:
            process = read(ROOT/'logs'/f'{ident}_bag_generate_process.json')
        controller = queue.get('pid', process.get('controller_pid'))
        replay = read(root/'previews'/ident/'independent_replay.json')
        visual = read(root/'previews'/ident/'asset_visual_review.json')
        result.append(dict(id=ident, root=str(root), stage=stage,
            state=row.get('state', process.get('state', 'prepared')),
            controller_pid=controller, controller_live=alive(controller),
            process_pid=process.get('pid'), process_live=alive(process.get('pid')),
            physical_gpu=process.get('physical_gpu') if alive(process.get('pid')) else None,
            asset_complete=replay.get('state')=='asset_only_replay_passed_quality_unreviewed',
            quality=visual.get('state', 'unreviewed'),
            quality_approved=visual.get('quality_approved', False),
            report=str(root/'REPORT.md'), scope='Separate accessory repair, not an extra identity or unchanged baseline'))
    return result


def publish():
    queue, live, people = collect()
    old = read(ROOT / 'status.json')
    old.update(state='background_queue_live' if live else 'queue_terminal_or_awaiting_review',
        updated_at=time.time(), controller_pid=queue.get('pid'), controller_live=live,
        selected_people=len(people), completed_assets=sum(p['asset_complete'] for p in people),
        generation_started=any((ROOT / 'logs' / f'{p["id"]}_generate_process.json').exists() for p in people),
        training_started=any((ROOT / 'logs' / f'{p["id"]}_native_process.json').exists() for p in people),
        generation_gate='Per-person actual input tensor/projection and visual approval required',
        people=people, results='RESULTS.csv')
    repairs = repair_status()
    any_live = live or any(r['controller_live'] for r in repairs)
    old['repair_branches'] = repairs
    old['any_background_controller_live'] = any_live
    old['state'] = 'background_queue_live' if any_live else 'queue_terminal_or_awaiting_review'
    old['first_two'] = {p['id']: p['state'] for p in people[:2]}
    atomic(ROOT / 'status.json', json.dumps(old, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    stream = io.StringIO()
    fields = list(people[0])
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k, v in p.items()} for p in people)
    atomic(ROOT / 'RESULTS.csv', stream.getvalue())
    return queue, any_live


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--continue-prepare', action='store_true')
    args = parser.parse_args()
    import fcntl
    lock = (ROOT / 'status_publisher.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        queue, live = publish()
        from publish_previews import gallery
        gallery()
        if not args.watch or not live:
            break
        await asyncio.sleep(30)
    # Publish the batch montage only after all eight fixed cases have a real
    # terminal outcome. Never produce an incomplete comparison during training.
    if (args.watch and not args.continue_prepare and
            queue.get('phase') == 'asset' and queue.get('state') == 'phase_finished'):
        from publish_previews import montage
        receipt = ROOT / 'gallery/montage_current_validation.json'
        if not receipt.exists():
            montage()
        gallery()
        from audit_delivery import main as audit_delivery
        audit_delivery()
    if args.continue_prepare:
        # Never restart an observed live process, a failure or an ambiguous exit.
        assert queue.get('state') == 'phase_finished' and queue.get('phase') == 'asset', queue
        assert not alive(queue['pid'])
        marker = ROOT / 'research/remaining_preparation_started.json'
        assert not marker.exists(), 'Remaining preparation already scheduled'
        atomic(ROOT / 'research/first_asset_queue.json', json.dumps(queue, indent=2) + '\n')
        ids = ['W03', 'W04', 'W05', 'W06', 'W07', 'W08']
        import sys
        with (ROOT / 'logs/remaining_preparation.log').open('a') as log:
            child = await asyncio.create_subprocess_exec(sys.executable, '-u',
                str(ROOT / 'scripts/run_people_queue.py'), '--phase', 'prepare', *ids,
                cwd=ROOT, stdout=log, stderr=asyncio.subprocess.STDOUT)
            atomic(marker, json.dumps(dict(pid=child.pid, ids=ids, started_at=time.time(),
                scope='Preparation only; source-pose review required before each new generation'), indent=2) + '\n')
            while child.returncode is None:
                publish()
                from publish_previews import gallery
                gallery()
                try:
                    await asyncio.wait_for(child.wait(), timeout=30)
                except asyncio.TimeoutError:
                    pass
            assert child.returncode == 0
        publish()
        gallery()


if __name__ == '__main__':
    asyncio.run(main())
