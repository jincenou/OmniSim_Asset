"""Compare only independently replayed terminal assets; preserve the original baseline."""
import argparse
import asyncio
from contextlib import ExitStack
import fcntl
import json
import os
import time

import av
from PIL import Image, ImageDraw
from prepare_person import ROOT, encode, save, sha
from publish_status import alive

PERSON = 'W02'
TRIAL = ROOT / f'diagnostics/accessory_{PERSON}_v1'


def publish():
    roots = (ROOT, TRIAL)
    evidence = []
    for root in roots:
        folder = root / 'previews' / PERSON
        report = json.loads((folder / 'independent_replay.json').read_text())
        assert report['state'] == 'asset_only_replay_passed_quality_unreviewed'
        assert report['source_reads'] == 0 and report['reference_max_abs_error'] < 1e-5
        media = json.loads((folder / 'media_validation.json').read_text())
        path = folder / 'asset_fixed_views.mp4'
        assert sha(path) == report['videos'][path.name]['sha256'] == media['files'][path.name]
        assert json.loads((root / 'assets' / PERSON / 'complete.json').read_text())['total_actual_updates'] == 52999
        evidence.append(dict(video=str(path), video_sha256=sha(path),
                             replay_report_sha256=sha(folder / 'independent_replay.json'),
                             camera_data_sha256=sha(root / 'assets' / PERSON / 'asset_camera_data.json')))
    plan = json.loads((TRIAL / 'GENERATION_PLAN.json').read_text())
    assert plan['camera_sha256'] == plan['baseline_camera_sha256']
    for root in roots:
        assert sha(root / 'frontend' / PERSON / 'preflight/motion/motion.safetensors') == plan['motion_tensors_sha256']
    assert sha(ROOT / 'backend/PLAN.json') == sha(TRIAL / 'backend/PLAN.json')

    def images():
        with ExitStack() as stack:
            containers = [stack.enter_context(av.open(x['video'])) for x in evidence]
            for container in containers:
                container.streams.video[0].thread_count = 1
            for index, batch in enumerate(zip(*(c.decode(video=0) for c in containers), strict=True)):
                canvas = Image.new('RGB', (1408, 1360), (128, 128, 128))
                for column, frame in enumerate(batch):
                    assert frame.width == 704 and frame.height == 1328
                    assert abs(float(frame.pts * frame.time_base) - index / 25) < 1e-8
                    label = 'ORIGINAL INPUT / FINAL ASSET' if column == 0 else 'PERSON + BAG INPUT / FINAL ASSET'
                    canvas.paste(frame.to_image(), (704 * column, 32))
                    ImageDraw.Draw(canvas).text((704 * column + 8, 10), label, fill='white')
                if index == 60:
                    canvas.save(TRIAL / 'asset_comparison.jpg', quality=94)
                yield canvas

    output = TRIAL / 'asset_comparison.mp4'
    encode(output, images())
    save(TRIAL / 'ASSET_COMPARISON.json', dict(
        at=time.time(), state='both_terminal_assets_replayed_comparison_quality_unreviewed',
        video_sha256=sha(output), image_sha256=sha(TRIAL / 'asset_comparison.jpg'),
        frames=121, fps=25, views=[0, 6, 12, 18], evidence=evidence,
        quality_approved=False, baseline_preserved=True,
        scope='Final Gaussian asset renders only; matched generated view IDs and source time. '
              'Each asset retains its own fitted coordinate normalization; no claim of identical normalized cameras.'))


async def main():
    global PERSON, TRIAL
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--person', choices=['W02','W04'], default='W02')
    args = parser.parse_args()
    PERSON=args.person;TRIAL=ROOT/f'diagnostics/accessory_{PERSON}_v1'
    lock = (TRIAL / 'asset_comparison.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (TRIAL / 'ASSET_COMPARISON.json').exists()
    if args.watch:
        while not (TRIAL / 'previews' / PERSON / 'media_validation.json').exists():
            queue = json.loads((TRIAL / 'queue_status.json').read_text())
            row = queue['people'][PERSON]
            if row['state'] == 'failed':
                raise RuntimeError('Repair backend failed; no asset comparison is published')
            if not alive(queue['pid']):
                raise RuntimeError('Repair controller exited without complete comparison inputs; inspect evidence')
            await asyncio.sleep(30)
    publish()
    print('Both terminal assets compared; visual quality review still required', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
