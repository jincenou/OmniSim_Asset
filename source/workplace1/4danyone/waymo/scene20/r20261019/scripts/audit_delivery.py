"""Audit existing delivered bytes; reuse all-frame decode evidence only with exact hashes."""
import asyncio
import collections
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
import av
import numpy as np

R = Path(__file__).resolve().parents[1]
def read(p):
    return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def decode(p, expected):
    with av.open(str(p)) as c:
        c.streams.video[0].thread_count = 1
        count = sum(1 for _ in c.decode(video=0))
    assert count == expected, (str(p), count, expected)
    return count

def sample(p):
    b = p.parent
    m = read(p)
    row = m['candidate']
    key, frames = row['sample_id'], m['frames']
    assert m['views'] == 24 and m['training'] is False
    assert frames == row['output_slots'] and frames in (61, 121)
    assert not (b / '.work').exists()
    assert read(b / 'candidate.json') == row
    actual = {str(x.relative_to(b)) for x in b.rglob('*') if x.is_file() and x != p}
    assert actual == set(m['retained']), (key, actual ^ set(m['retained']))
    for rel, digest in m['retained'].items():
        assert sha(b / rel) == digest, (key, rel)
    native = read(b / 'input/native.json')
    mapping = read(b / 'input/frame_mapping.json')
    assert len(native['frames']) == row['native_frame_count']
    assert len(mapping['frames']) == frames
    assert mapping['endpoint_padding'] is False and mapping['speedup'] is False
    assert mapping['max_time_regularization_s'] <= .001
    assert mapping['native_source_sha256'] == sha(b / 'input/native.json')
    for f in native['frames']:
        assert sha(Path(f['raw_source_path'])) == f['raw_source_sha256']
    assert sha(Path(native['source_camera_audit'])) == native['source_camera_audit_sha256']
    for rel in ('root_motion.npz', 'input/native_cameras.npz', 'input/source_cameras.npz', 'input/target_masks.npz'):
        with np.load(b / rel, allow_pickle=False) as data:
            assert all(np.isfinite(data[k]).all() for k in data.files), (key, rel)
            if rel == 'root_motion.npz':
                assert data['frame_times'].shape == (frames,)
                assert np.allclose(data['frame_times'], np.arange(frames) / 25)
                assert np.allclose(data['local_to_canonical'] @ data['canonical_to_local'], np.eye(4), atol=1e-6)
            if rel == 'input/target_masks.npz':
                assert data['masks'].shape[0] == frames
                assert (data['masks'].reshape(frames, -1).sum(1) > 0).all()
    cameras = read(b / 'cameras.json')
    assert cameras['world_frame']['name'] == 'root_local_human_world'
    assert len(cameras['cameras']) == 24
    assert sha(b / cameras['root_motion']['file']) == cameras['root_motion']['sha256']
    validation = read(b / 'validation.json')
    assert validation['state'] == 'all24_videos_decoded_same_retained_cameras'
    assert len(validation['videos']) == 24
    assert {x.name for x in (b / 'videos').iterdir()} == {f'{v:02d}.mp4' for v in range(24)}
    for v, (cam, check) in enumerate(zip(cameras['cameras'], validation['videos'], strict=True)):
        assert cam['camera_id'] == check['view'] == v
        assert (cam['image_width'], cam['image_height']) == (704, 1280)
        assert check['frames'] == frames
        assert check['sha256'] == m['retained'][cam['video']]
        assert np.isfinite(cam['K']).all() and np.isfinite(cam['camera_to_world']).all()
        assert np.isclose(np.linalg.det(np.asarray(cam['camera_to_world'])[:3, :3]), 1, atol=1e-5)
        decode(b / cam['skeleton_video'], frames)
    for rel, n in [('preview.mp4', frames), ('input/model_input.mp4', frames), ('input/observed_native.mp4', row['native_frame_count'])]:
        decode(b / rel, n)
    generation = read(b / 'generation.json')
    assert generation['generation']['seed'] == 42
    assert generation['generation']['inference_steps'] == 4
    assert generation['output']['target_views'] == 24
    stages = ('camera', 'crop', 'interpolate', 'seg_prepare', 'segment', 'composite', 'motion', 'motion_gate', 'conditioning', 'generate', 'deliver')
    receipts = [read(R / 'research' / key / f'{stage}.json') for stage in stages]
    assert len(receipts) == 11 and all(x['exit_code'] == 0 for x in receipts)
    return {'sample': key, 'scene': row['scene'], 'identity_key': row['identity_key'],
            'frames': frames, 'files_hashed': len(actual), 'original_videos_verified': 24,
            'original_decode_evidence': 'All frames and PTS checked in deliver.finish; exact delivered SHA256 reverified here',
            'skeletons_and_input_preview_decoded_again': True, 'manifest_sha256': sha(p),
            'path': str(b.relative_to(R)), 'generation_seconds': generation['generation']['total_elapsed_seconds'],
            'gpu': next(x['gpu'] for x in receipts if x['stage'] == 'generate'),
            'bytes': sum(x.stat().st_size for x in b.rglob('*') if x.is_file())}

async def main():
    sem = asyncio.Semaphore(3)
    async def checked(p):
        async with sem:
            key = p.parent.name
            saved = R / f'research/{key}/delivery_audit.json'
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            with (R / f'logs/{key}_delivery_audit.log').open('w') as log:
                proc = await asyncio.create_subprocess_exec(sys.executable, __file__, '--sample', str(p), env=env, stdout=log, stderr=log)
                rc = await proc.wait()
            assert rc == 0, (key, rc)
            result = read(saved)
            print(json.dumps({'validated': result['sample']}), flush=True)
            return result
    rows = await asyncio.gather(*(checked(p) for p in sorted((R / 'scenes').glob('*/*/manifest.json'))))
    assert len({r['identity_key'] for r in rows}) == len(rows)
    assert len({r['scene'] for r in rows}) == len(rows)
    jobs = {}
    for name in ('STATUS.json', 'RESERVE_STATUS.json'):
        jobs.update(read(R / name)['jobs'])
    assert {r['sample'] for r in rows} == {k for k, v in jobs.items() if v['state'] == 'complete'}
    output = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'target': 20, 'completed_people': len(rows), 'execution_target_verified': len(rows) == 20,
              'quality_approval': False, 'quality_scope': 'Technical delivery only; see visual review, no real novel-view truth',
              'attempts': len(jobs), 'job_states': dict(collections.Counter(x['state'] for x in jobs.values())),
              'frame_counts': dict(collections.Counter(x['frames'] for x in rows)), 'samples': rows}
    (R / 'research/delivery_audit.json').write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in output.items() if k != 'samples'}), flush=True)

if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--sample':
        p = Path(sys.argv[2])
        result = sample(p)
        (R / f'research/{p.parent.name}/delivery_audit.json').write_text(json.dumps(result, indent=2))
    else:
        asyncio.run(main())
