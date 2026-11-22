"""Verify a terminal accessory trial and publish its links without claiming quality."""
import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import time

from audit_delivery import ROOT, Links, read, sha, video
from publish_status import alive, publish
from publish_previews import gallery


def audit(person):
    trial = ROOT / 'diagnostics' / f'accessory_{person}_v1'
    asset = trial / 'assets' / person
    preview = trial / 'previews' / person
    package = read(asset / 'package_manifest.json')
    assert package['actual_updates'] == 52999
    assert read(asset / 'complete.json')['total_actual_updates'] == 52999
    assert os.access(asset / 'replay.sh', os.X_OK)
    for name, digest in package['files'].items():
        path = asset / name
        assert path.resolve().is_relative_to(asset.resolve())
        assert sha(path) == digest, path
    replay = read(preview / 'independent_replay.json')
    assert replay['state'] == 'asset_only_replay_passed_quality_unreviewed'
    assert replay['source_reads'] == 0 and replay['reference_count'] == 12
    assert replay['noninteger_reference_time'] == 12.25
    assert replay['reference_max_abs_error'] < 1e-5
    assert replay['rendered_foreground_min'] > 0
    assert replay['orbit_camera_ring_max_abs_error'] < 1e-5
    media = read(preview / 'media_validation.json')
    assert media['source_and_asset_camera_aligned'] is False
    for name, digest in media['files'].items():
        assert sha(preview / name) == digest, name
    videos = {}
    for name in ('asset_fixed_views.mp4', 'asset_orbit.mp4',
                 'asset_view00.mp4', 'source_vs_asset.mp4'):
        videos[name] = video(preview / name)
        if name in replay['videos']:
            assert videos[name]['sha256'] == replay['videos'][name]['sha256']
    comparison = read(trial / 'ASSET_COMPARISON.json')
    assert comparison['state'] == 'both_terminal_assets_replayed_comparison_quality_unreviewed'
    assert comparison['baseline_preserved'] is True
    assert sha(trial / 'asset_comparison.jpg') == comparison['image_sha256']
    videos['asset_comparison.mp4'] = video(trial / 'asset_comparison.mp4')
    assert videos['asset_comparison.mp4']['sha256'] == comparison['video_sha256']
    for evidence in comparison['evidence']:
        assert sha(Path(evidence['video'])) == evidence['video_sha256']
    report = dict(at=time.time(), state='terminal_integrity_passed_visual_review_required',
                  person=person, asset_files=len(package['files']), videos=videos,
                  manifest_sha256=sha(asset / 'package_manifest.json'),
                  replay_sha256=sha(preview / 'independent_replay.json'),
                  comparison_sha256=sha(trial / 'ASSET_COMPARISON.json'),
                  quality_approved=False,
                  scope='File integrity, complete media decoding and existing independent replay evidence; no new quality approval.')
    target = trial / 'research/delivery_integrity.json'
    target.parent.mkdir(exist_ok=True)
    temporary = target.with_suffix('.writing')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(target)
    return report


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--person', choices=['W02', 'W04'], required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--comparison-pid', type=int)
    args = parser.parse_args()
    trial = ROOT / 'diagnostics' / f'accessory_{args.person}_v1'
    lock = (trial / 'delivery_audit.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.watch:
        while not (trial / 'ASSET_COMPARISON.json').exists():
            queue = read(trial / 'queue_status.json')
            assert queue['people'][args.person]['state'] != 'failed', queue
            assert alive(queue['pid']) or alive(args.comparison_pid), 'No live backend/comparison producer; inspect failure, do not publish'
            await asyncio.sleep(30)
    report = audit(args.person)
    # Serialize with the existing publisher; its last refresh may precede the
    # independent comparison watcher. Never stop training or another publisher.
    publication = (ROOT / 'status_publisher.lock').open('a+')
    while True:
        try:
            fcntl.flock(publication.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            await asyncio.sleep(10)
    publish()
    gallery()
    page = ROOT / 'gallery/index.html'
    links = Links()
    links.feed(page.read_text())
    required = f'../diagnostics/accessory_{args.person}_v1/asset_comparison.mp4'
    assert required in links.links, 'Finished comparison missing from gallery'
    assert all((page.parent / link).is_file() for link in links.links)
    print(json.dumps(dict(state=report['state'], person=args.person,
                          asset_files=report['asset_files'], gallery_links=len(links.links))), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
