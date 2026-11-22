"""Verify current delivered assets and links; pending work is never a pass."""
import csv
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import time
from urllib.parse import unquote, urlsplit

import av

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def video(path):
    count = 0
    with av.open(str(path)) as container:
        container.streams.video[0].thread_count = 1
        for i, frame in enumerate(container.decode(video=0)):
            assert abs(float(frame.pts * frame.time_base) - i / 25) < 1e-7, path
            count += 1
    assert count == 121, (path, count)
    return dict(frames=count, fps=25, sha256=sha(path))


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = set()

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('href', 'src') and value:
                parsed = urlsplit(value)
                if not parsed.scheme and parsed.path:
                    self.links.add(unquote(parsed.path))


def main():
    from publish_status import collect
    queue, live, people = collect()
    frozen = read(ROOT / 'SELECTED_TRACKS.json')['tracks']
    ids = [p['id'] for p in frozen]
    assert len(ids) == len(set(ids)) == 8
    assert [p['id'] for p in people] == ids
    with (ROOT / 'RESULTS.csv').open() as stream:
        csv_ids = [r['id'] for r in csv.DictReader(stream)]
    assert csv_ids == ids
    page = (ROOT / 'gallery/index.html').read_text()
    assert all(f'<h2>{ident} ' in page for ident in ids)
    parser = Links()
    parser.feed(page)
    missing_links = [link for link in sorted(parser.links) if not (ROOT / 'gallery' / link).is_file()]
    assert not missing_links, missing_links
    rows = []
    for person in people:
        ident = person['id']
        row = dict(id=ident, state=person['state'], quality=person['quality'],
                   source=video(ROOT / 'inputs' / ident / 'source_clip.mp4'))
        if person['asset_complete']:
            required_media = {
                'source_crop': ROOT / 'inputs' / ident / 'source_crop.mp4',
                'pose_preview': ROOT / 'inputs' / ident / 'pose_preview.mp4',
                'generated_multiview': ROOT / 'previews' / ident / 'generated_multiview.mp4',
            }
            row['upstream_media'] = {name: video(path) for name, path in required_media.items()}
            generation = read(ROOT / 'frontend' / ident / 'research/generated_validation.json')
            assert generation['state'] == 'all24_videos_decoded_same_retained_cameras'
            assert [item['view'] for item in generation['videos']] == list(range(24))
            for item in generation['videos']:
                path = ROOT / 'frontend' / ident / 'outputs/turbo/videos' / f"{item['view']:02d}.mp4"
                assert item['frames'] == 121 and sha(path) == item['sha256'], path
            row['generated_original_videos_hash_verified'] = 24
            masks = read(ROOT / 'backend/research' / f'{ident}-mask-ready-complete.json')
            assert masks['state'] == '24_view_RGBA_complete_not_quality_approved'
            assert masks['views'] == 24 and masks['frames'] == 121
            assert [item['view'] for item in masks['alpha_checks']] == list(range(24))
            row['rgba_validation_report_sha256'] = sha(ROOT / 'backend/research' / f'{ident}-mask-ready-complete.json')
            asset = ROOT / 'assets' / ident
            package = read(asset / 'package_manifest.json')
            assert package['actual_updates'] == 52999
            assert read(asset / 'complete.json')['total_actual_updates'] == 52999
            assert os.access(asset / 'replay.sh', os.X_OK)
            for name, digest in package['files'].items():
                path = asset / name
                assert path.resolve().is_relative_to(asset.resolve())
                assert sha(path) == digest, path
            preview = ROOT / 'previews' / ident
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
            for name in ('asset_fixed_views.mp4', 'asset_orbit.mp4', 'asset_view00.mp4', 'source_vs_asset.mp4'):
                videos[name] = video(preview / name)
                if name in replay['videos']:
                    assert videos[name]['sha256'] == replay['videos'][name]['sha256']
            row.update(asset=str(asset), manifest_sha256=sha(asset / 'package_manifest.json'),
                       verified_asset_files=len(package['files']), videos=videos,
                       replay_report_sha256=sha(preview / 'independent_replay.json'),
                       quality_approved=False,
                       limitations='Runtime-dependent dynamic asset; source-action replay only. No verified world scale, hidden-view ground truth or arbitrary motion driving.')
        rows.append(row)
    terminal = {'asset_replayed_quality_unreviewed', 'segmentation_gate_failed', 'stage_failed', 'camera_interface_not_applicable_pending'}
    pending = [p['id'] for p in people if p['state'] not in terminal]
    montage = ROOT / 'gallery/montage_validation.json'
    montage_check = None
    if montage.exists():
        assert not live and not pending
        proof = read(montage)
        assert [p['id'] for p in proof['people']] == ids
        montage_check = video(ROOT / 'gallery/montage.mp4')
        assert montage_check['sha256'] == proof['sha256']
    report = dict(at=time.time(), state='all_eight_outcomes_packaged' if not pending and not live and montage_check else 'incomplete_background_work',
                  queue_live_at_start=live, pending_ids=pending,
                  selected_sha256=sha(ROOT / 'SELECTED_TRACKS.json'),
                  verified_local_gallery_links=len(parser.links), people=rows, montage=montage_check,
                  scope='Delivery integrity and source-independent replay evidence; not blanket visual quality approval and not completion of the separate Waymo pool task.')
    path = ROOT / 'research/delivery_audit.json'
    tmp = path.with_suffix('.writing')
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)
    print(json.dumps(dict(state=report['state'], pending=pending,
                          verified_assets=sum('asset' in r for r in rows), links=len(parser.links))))


if __name__ == '__main__':
    main()
