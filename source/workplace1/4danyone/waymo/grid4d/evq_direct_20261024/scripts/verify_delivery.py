"""Read-only delivery/protocol audit. Does not select a model or alter training."""
import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import time
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent.parent / 'random8/r20261015_113332'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    temporary = path.with_suffix('.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def signature(metric):
    return [(r['view'], r['frame'], r['seconds'], r['crop_xyxy'], r['input_sha256']) for r in metric['samples']]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        self.paths.extend(v for k, v in attrs if k in ('src', 'href'))


async def main(wait):
    if wait:
        while not (ROOT / 'queue_exit.json').exists():
            status = read(ROOT / 'STATUS.json')
            assert status['failed'] == 0, 'Queue failure requires explicit repair'
            os.kill(status['controller_pid'], 0)
            await asyncio.sleep(30)
    assert read(ROOT / 'queue_exit.json')['post_exit_code'] == 0
    assert read(ROOT / 'post.complete.json')['numeric_media_complete']
    plan = read(ROOT / 'PLAN.json')
    history = read(ROOT / 'queue_history.json')
    proofs, costs = [], []
    for label in plan['labels']:
        a = read(plan['inputs'][label]['old_references']['A']['metrics'])
        sig = signature(a)
        assert len(sig) == 39 and len({(x[0], x[1]) for x in sig}) == 39
        for view, frame, seconds, crop, digest in sig:
            assert view in (0, 8, 16) and seconds == frame / 25
            reference = BASE / 'samples' / label / 'data/images' / f'cam{view:02d}' / f'{frame:06d}.png'
            assert hashlib.sha256(reference.read_bytes()).hexdigest() == digest
        methods = {'A': a}
        for method in ['R0', 'R3'] + plan['arms'] + ([] if label == 'N04' else ['G2']):
            parent = ROOT if method in plan['arms'] else ROOT.parent / ('evq_primitive_local_20261019' if method == 'G2' else 'early_visual_query_20261019')
            methods[method] = read(parent / 'results' / f'{label}_{method}_32999' / 'metrics.json')
        for method, metric in methods.items():
            assert signature(metric) == sig, (label, method, 'Reference/crop/time mismatch')
            for field, value in metric['mean'].items():
                assert math.isfinite(value)
                measured = statistics.mean(r[field] for r in metric['samples'])
                assert abs(measured - value) < 1e-9, (label, method, field)
            proofs.append(dict(label=label, method=method, observations=39, exact_reference_crops_times=True, means_recomputed=True))
        for arm in plan['arms']:
            key = f'{label}_{arm}'
            stages = [r for r in history if r['key'] == key]
            expected = {'train', 'export', 'replay', 'evaluate', 'parts'} | (set() if label == 'N04' else {'expanded'})
            assert len(stages) == len(expected) and {r['stage'] for r in stages} == expected
            assert all(r['exit_code'] == 0 and Path(r['marker']).is_file() for r in stages)
            assert len({r['gpu']['uuid'] for r in stages}) == 1
            done = read(ROOT / 'runs' / key / 'training.complete.json')
            counts = done['counts']
            assert counts['additional_updates'] == 30000 and counts['logical_terminal'] == 32999
            record = dict(key=key, physical_gpu=stages[0]['gpu']['index'], gpu_uuid=stages[0]['gpu']['uuid'], points=done['points'], logical_updates=32999, new_updates=30000, imported_warmup_updates=2999, imported_warmup_wall_seconds=None,
                          stage_subprocess_seconds={r['stage']: r['seconds'] for r in stages}, active_stage_sum_seconds=sum(r['seconds'] for r in stages),
                          training_entry_seconds=done['seconds'], training_loop_seconds=done['loop_seconds'], training_load_seconds=done['load_seconds'],
                          new_training_renders=counts['target_renders'] + counts['alpha_renders'] + counts['visibility_renders'], visibility_renders=counts['visibility_renders'])
            costs.append(record)
    parser = Links()
    parser.feed((ROOT / 'comparison/index.html').read_text())
    for value in parser.paths:
        assert '://' not in value and (ROOT / 'comparison' / value).resolve().is_file(), value
    videos = []
    for label in plan['labels']:
        receipt = read(ROOT / 'comparison' / label / 'complete.json')
        assert receipt['fully_decoded'] and len(receipt['files']) == 3
        for video in receipt['files']:
            assert (ROOT / video['path']).is_file() and video['decoded_frames'] == plan['inputs'][label]['frame_count']
            assert all(arm in video['columns'] for arm in plan['arms'])
            videos.append(video)
    save(ROOT / 'research/delivery_audit.json', dict(state='protocol_artifacts_and_costs_verified', at=time.time(), checks=proofs, videos=videos, local_html_links=len(parser.paths), visual_quality_verified=False, default_changed=False))
    save(ROOT / 'cost_summary.json', dict(state='all_12_stages_accounted', records=costs, total_new_updates=sum(x['new_updates'] for x in costs),
        scope='Stage subprocess wall time includes interpreter/startup. Entry time excludes imports; loop includes checkpoint saves. Stage sums exclude waiting, prior shared warmup, frontend generation and final comparison media. Concurrent stage sums are not calendar duration or pure GPU compute time.',
        shared_load_not_fair_speed_ranking=True, same_device_controlled_speedup_measured=False, default_changed=False))
    print('Delivery audit: exact reference/crop/time and means verified; 12 stage histories; 9 decoded final comparisons.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true')
    asyncio.run(main(parser.parse_args().wait))
