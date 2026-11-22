"""Audit bounded, labeled Waymo candidates; preserve source geometry and time."""
import asyncio
import hashlib
import json
import sys
import os
from pathlib import Path
from fractions import Fraction

import av
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/mnt/public_data/waymo/train')
CAM = {'FRONT': 0, 'FRONT_LEFT': 1, 'FRONT_RIGHT': 2}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def encode(path, images, fps):
    with av.open(str(path), 'w') as out:
        stream = out.add_stream('libx264rgb', rate=fps)
        stream.width, stream.height = images[0].size
        stream.pix_fmt = 'rgb24'
        stream.thread_count = 1
        stream.options = {'crf': '0', 'preset': 'fast', 'threads': '1'}
        for idx, im in enumerate(images):
            f = av.VideoFrame.from_image(im)
            f.pts = idx
            f.time_base = Fraction(1, fps)
            for packet in stream.encode(f):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def candidate_preview(row):
    cid = row['candidate_id']
    camera = CAM[row['camera']]
    obs = row['observations']
    bounds = np.array([[o['bbox_2d'][k] for k in ('xmin', 'ymin', 'xmax', 'ymax')] for o in obs])
    lo = np.floor(bounds[:, :2].min(0) - 30).astype(int)
    hi = np.ceil(bounds[:, 2:].max(0) + 30).astype(int)
    canvas = []
    hashes = []
    for o in obs:
        path = DATA / row['scene'] / 'images' / f"{o['frame_index']:03d}_{camera}.jpg"
        im = Image.open(path).convert('RGB')
        hashes.append(hashlib.sha256(im.tobytes()).hexdigest())
        draw = ImageDraw.Draw(im)
        b = o['bbox_2d']
        draw.rectangle(tuple(b[k] for k in ('xmin', 'ymin', 'xmax', 'ymax')), outline=(0, 255, 0), width=3)
        preview = Image.new('RGB', (640, 400), (25, 25, 25))
        whole = ImageOps.contain(im, (400, 340))
        close = ImageOps.contain(im.crop((*lo, *hi)), (232, 340))
        preview.paste(whole, (0, 45))
        preview.paste(close, (404, 45))
        ImageDraw.Draw(preview).text((8, 8), f"{cid} scene {row['scene']} {row['camera']} frame {o['frame_index']} | native 10 Hz", fill='white')
        canvas.append(preview)
    encode(ROOT / 'previews' / f'{cid}.mp4', canvas, 10)
    row['preview'] = f'previews/{cid}.mp4'
    row['decoded_duplicate_frames'] = len(hashes) - len(set(hashes))
    row['source_dimensions'] = [1920, 1280]
    row['fps_nominal'] = 10
    row['timestamp_step_us_range'] = [int(v) for v in np.percentile(np.diff([o['timestamp_micros'] for o in obs]), [0, 100])]
    row['camera_index'] = camera
    row['occlusion_status'] = 'requires visual review; difficulty labels are not occlusion fractions'
    write_json(ROOT / 'research' / f'{cid}.json', row)
    return row


async def main():
    rows = json.loads((ROOT / 'candidate_manifest.json').read_text())
    candidate_id = os.environ.get('WAYMO_MAIN', 'C01')
    target = next(row for row in rows if row['candidate_id']==candidate_id)
    obs = target['observations'][-121:]
    crop = {'C00':[850,564,1138,1076],'C01':[1090,640,1314,1024]}[candidate_id]
    Kraw = np.loadtxt(DATA / '001/intrinsics/1.txt')
    K = np.array([[Kraw[0], 0, Kraw[2]], [0, Kraw[1], Kraw[3]], [0, 0, 1]])
    A = np.array([[1, 0, -crop[0]], [0, 1, -crop[1]], [0, 0, 1]])
    frames, records, hashes, poses = [], [], [], []
    for i, o in enumerate(obs):
        source = DATA / '001/images' / f"{o['frame_index']:03d}_1.jpg"
        im = Image.open(source).convert('RGB')
        frames.append(im.crop(crop))
        hashes.append(hashlib.sha256(im.tobytes()).hexdigest())
        pose = np.loadtxt(DATA / '001/ego_to_world' / f"{o['frame_index']:03d}.txt")
        poses.append(pose)
        b = o['bbox_2d']
        records.append({'input_frame': i, 'model_slot': i, 'frame_index': o['frame_index'], 'source_path': str(source),
                        'timestamp_micros': o['timestamp_micros'], 'relative_seconds': (o['timestamp_micros'] - obs[0]['timestamp_micros']) / 1e6,
                        'video_seconds': i / 10, 'crop_xyxy': crop, 'source_to_input': A.tolist(), 'K_input': (A @ K).tolist(),
                        'bbox_input_xyxy': [b['xmin']-crop[0], b['ymin']-crop[1], b['xmax']-crop[0], b['ymax']-crop[1]],
                        'ego_to_world': pose.tolist(), 'source_camera_to_world_file': str(DATA / '001/cam_to_world' / f"{o['frame_index']:03d}_1.txt"),
                        'interpolated': False, 'enhanced': False})
    encode(ROOT / 'inputs/main_10fps.mp4', frames, 10)
    poses = np.stack(poses)
    rotations = poses[:, :3, :3] @ poses[0, :3, :3].T
    angles = np.degrees(np.arccos(np.clip((np.trace(rotations, axis1=1, axis2=2)-1)/2, -1, 1)))
    manifest = {'candidate_id': candidate_id, 'scene': '001', 'camera': 'FRONT_LEFT', 'camera_index': 1, 'track_id': target['track'],
                'selection': 'last 121 frames, full-body label, near-stationary ego, fixed crop isolates main pedestrian',
                'source_intrinsic': K.tolist(), 'source_distortion': Kraw[4:].tolist(), 'undistorted': False,
                'camera_axes': 'Converted cam_to_world convention not assumed; source matrices retained, not consumed by static GVHMR.',
                'true_calibration_used_by_gvhmr': False, 'frames': records}
    write_json(ROOT / 'inputs/frame_mapping.json', manifest)
    audit = {'status': 'input_prepared_motion_preflight_pending', 'bounded_scope': '10 pre-exported training scenes; not exhaustive Waymo search',
             'candidate_count': len(rows), 'continuous_runs_ge49': len(json.loads((ROOT/'research/track_scan.json').read_text())),
             'data_formats': ['converted RGB JPEG + calibration/poses', 'camera-ID track JSON', 'raw TFRecord available', '3DSkelMo NPZ available, not required'],
             'main': {k:v for k,v in manifest.items() if k!='frames'}, 'frames':121, 'native_fps':10,
             'true_span_seconds': records[-1]['relative_seconds'], 'encoded_span_seconds':12,
             'max_timestamp_regularization_seconds': max(abs(x['relative_seconds']-x['video_seconds']) for x in records),
             'duplicate_source_frames': len(hashes)-len(set(hashes)),
             'ego_translation_path_m':float(np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1).sum()),
             'ego_net_displacement_m':float(np.linalg.norm(poses[-1,:3,3]-poses[0,:3,3])), 'ego_rotation_max_deg':float(angles.max()),
             'body_pixels':dict(zip(('min','median','max'),map(float,np.percentile([o['bbox_2d']['ymax']-o['bbox_2d']['ymin'] for o in obs],[0,50,100])))),
             'preprocessing':f'fixed {crop[2]-crop[0]}x{crop[3]-crop[1]} crop, no resize, no interpolation, no enhancement, actual source timestamps retained',
             'limitations':['Main subject mostly stationary: this does not test walking-cycle recovery.', 'No instance masks provided by chosen export.', 'Real camera intrinsics stored but official motion wrapper estimates its own.']}
    write_json(ROOT / 'input_audit.json', audit)
    print(json.dumps(audit), flush=True)
    sem = asyncio.Semaphore(3)
    async def render(row):
        async with sem:
            record=ROOT/'research'/f"{row['candidate_id']}.json"
            if record.is_file() and (ROOT/'previews'/f"{row['candidate_id']}.mp4").is_file():
                return json.loads(record.read_text())
            proc = await asyncio.create_subprocess_exec(sys.executable, __file__, row['candidate_id'])
            code = await proc.wait()
            if code:
                raise RuntimeError(f"Candidate preview failed: {row['candidate_id']}, exit {code}")
            return json.loads((ROOT/'research'/f"{row['candidate_id']}.json").read_text())
    rows = await asyncio.gather(*map(render, rows))
    write_json(ROOT / 'candidate_manifest.json', rows)


if __name__ == '__main__':
    if len(sys.argv)==2:
        rows=json.loads((ROOT/'candidate_manifest.json').read_text())
        candidate_preview(next(row for row in rows if row['candidate_id']==sys.argv[1]))
    else:
        asyncio.run(main())
