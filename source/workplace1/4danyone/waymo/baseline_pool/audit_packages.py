"""Read-only package integrity audit; never promotes numerical checks to quality."""
import asyncio
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from safetensors.numpy import load_file
from package_catalog import ready_paths
from scan_native import ROOT, write


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check(path):
    start = time.monotonic()
    directory = path.parent
    ready = json.loads(path.read_text())
    assert ready['generation_started'] is False
    assert ready['quality_accepted'] is False
    candidate = ready['candidate']
    assert candidate == json.loads((directory / 'candidate.json').read_text())
    count = candidate['output_slots']
    assert count in (61, 121)
    verified_bytes = 0
    paths = set()
    for video in ready['videos']:
        file = directory / video['path']
        assert file.resolve().is_relative_to(directory.resolve()), str(file)
        assert video['path'] not in paths
        paths.add(video['path'])
        assert sha(file) == video['sha256'], f'Changed decoded media: {file}'
        verified_bytes += file.stat().st_size
        expected = (count - 1) * 2 // 5 + 1 if video['path'] == 'clip0/observed_native.mp4' else count
        assert video['frames'] == expected, (file, video['frames'], expected)
    assert len(paths) >= 28 and ready['inputs'] in paths and ready['input_skeleton_comparison'] in paths
    clip = directory / 'clip0'
    mapping_path = clip / 'inputs/frame_mapping.json'
    mapping = json.loads(mapping_path.read_text())
    frames = mapping['frames']
    assert len(frames) == count and [f['slot'] for f in frames] == list(range(count))
    expected_clock = np.arange(count) / 25
    assert np.allclose([f['video_seconds'] for f in frames], expected_clock, atol=1e-8, rtol=0)
    assert not mapping['speedup'] and not mapping['endpoint_padding']
    assert 0 < mapping['source_speed_ratio'] <= 1 + 1e-9
    camera_path = clip / 'inputs/cameras.npz'
    assert sha(camera_path) == mapping['camera_npz_sha256']
    assert sha(clip / 'native.json') == mapping['native_source_sha256']
    with np.load(camera_path, allow_pickle=False) as arrays:
        assert np.allclose(arrays['K_input'], np.asarray([f['K_input'] for f in frames]), atol=1e-6, rtol=0)
        assert np.allclose(arrays['video_seconds'], expected_clock, atol=1e-8, rtol=0)
        assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
    motion = directory / ready['motion']
    info = json.loads((motion / 'motion.json').read_text())
    assert info['num_frames'] == count and info['fps_num'] / info['fps_den'] == 25
    assert np.allclose(info['frame_timestamps_sec'], expected_clock, atol=1e-8, rtol=0)
    tensors = load_file(motion / 'motion.safetensors')
    assert tensors and all(np.isfinite(x).all() for x in tensors.values())
    with np.load(motion / 'projection.npz', allow_pickle=False) as projection:
        assert all(np.isfinite(projection[k]).all() for k in projection.files)
        assert projection['keypoints_uv'].shape[0] == count
    condition = directory / ready['conditioning']
    meta = json.loads((condition / 'metadata.json').read_text())
    assert meta['num_frames'] == count and meta['fps_num'] / meta['fps_den'] == 25
    points = np.load(condition / 'keypoints_3d.npy', allow_pickle=False)
    assert points.shape == (count, len(meta['keypoint_names']), 3) and np.isfinite(points).all()
    cameras = json.loads((condition / 'cameras.json').read_text())['cameras']
    assert len(cameras) == 24 and {x['camera_id'] for x in cameras} == set(range(24))
    for camera in cameras:
        a = np.asarray(camera['world_to_camera'])
        b = np.asarray(camera['camera_to_world'])
        k = np.asarray(camera['K'])
        assert np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(k).all()
        assert np.allclose(a @ b, np.eye(4), atol=1e-5, rtol=0)
    gate = json.loads((condition.parent / 'review.json').read_text())
    assert gate['allow_generation'] is False
    audit = directory / 'skeleton_audit.json'
    review = directory / 'review.json'
    return dict(package=str(directory.relative_to(ROOT)), state='integrity_passed_quality_pending',
        ready_sha256=sha(path), mapping_sha256=sha(mapping_path), verified_video_count=len(paths),
        verified_video_bytes=verified_bytes, frames=count, identity_scope=candidate['identity_scope'],
        source_speed_ratio=mapping['source_speed_ratio'],
        skeleton_diagnostic_present=audit.is_file(), visual_review_present=review.is_file(),
        seconds=time.monotonic()-start, quality_accepted=False)


async def main():
    folder = ROOT / 'research/package_integrity'
    folder.mkdir(exist_ok=True)
    lock = (folder / 'audit.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    paths = ready_paths()
    start = time.time()
    state = dict(state='running', pid=os.getpid(), started_at=start, total=len(paths), checked=0,
                 failed=0, verified_video_bytes=0, quality_accepted_people=None,
                 script_sha256=sha(Path(__file__)))
    write(folder / 'status.json', state)
    semaphore = asyncio.Semaphore(4)
    rows = []
    async def one(path):
        async with semaphore:
            try:
                row = await asyncio.to_thread(check, path)
            except Exception as exc:
                row = dict(package=str(path.parent.relative_to(ROOT)), state='integrity_failed', error=repr(exc))
            rows.append(row)
            state['checked'] += 1
            state['failed'] += row['state'] == 'integrity_failed'
            state['verified_video_bytes'] += row.get('verified_video_bytes', 0)
            if state['checked'] % 50 == 0 or state['checked'] == len(paths):
                state['updated_at'] = time.time()
                write(folder / 'status.json', state)
    await asyncio.gather(*(one(path) for path in paths))
    assert ready_paths() == paths, 'Package set changed while auditing'
    write(folder / 'results.json', sorted(rows, key=lambda row: row['package']))
    state.update(state='finished_with_integrity_failures' if state['failed'] else 'integrity_passed_quality_pending',
                 finished_at=time.time(), seconds=time.time()-start,
                 verified_video_count=sum(row.get('verified_video_count', 0) for row in rows),
                 identity_scope_counts=dict(collections.Counter(row.get('identity_scope','failed') for row in rows)),
                 visual_review_present=sum(row.get('visual_review_present', False) for row in rows),
                 results_sha256=sha(folder / 'results.json'),
                 scope='All READY media SHA256 matches earlier complete decoding receipts; current numeric/time/camera checks. No new RGB generation, no pose-quality approval, no identity merging. Source JPEG contents and perceptual video quality not audited here.')
    write(folder / 'status.json', state)
    print(json.dumps(state), flush=True)
    if state['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    asyncio.run(main())
