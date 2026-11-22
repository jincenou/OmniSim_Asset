"""Read-only P18 stage audit; no model inference or changes to delivered assets."""
from pathlib import Path
import json
import hashlib
import csv
import argparse
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from safetensors import safe_open

OUT = Path(__file__).resolve().parent
ARGS = argparse.ArgumentParser()
ARGS.add_argument('--sample', default='005/P18')
SAMPLE = ARGS.parse_args().sample
SRC = OUT.parent / 'scene20/r20261019/scenes' / SAMPLE
NAME = SRC.name
SLOTS = [0, 15, 30, 45, 60]
FONT = ImageFont.load_default(size=15)

def read_video(path, keep):
    result = {}
    with av.open(str(path)) as container:
        container.streams.video[0].thread_count = 1
        for i, frame in enumerate(container.decode(video=0)):
            if i in keep:
                result[i] = frame.to_image()
    return result, i + 1

def main():
    mapping = json.loads((SRC/'input/frame_mapping.json').read_text())['frames']
    nearest = {s: mapping[s]['left_native_index'] if mapping[s]['interpolation_fraction'] < .5
               else mapping[s]['right_native_index'] for s in range(61)}
    inputs = [('Native (nearest exposure)', 'input/observed_native.mp4'),
              ('Actual masked input', 'input/model_input.mp4'),
              ('GVHMR: green / detector: red', 'motion/source_fit.mp4'),
              ('Skeleton condition view 04', 'skeletons/04.mp4'),
              ('Generated view 04', 'videos/04.mp4')]
    cache = []
    counts = {}
    hashes = {}
    for label, relative in inputs:
        path = SRC/relative
        frames, count = read_video(path, set(nearest.values()) if relative.endswith('observed_native.mp4') else set(range(61)))
        counts[relative] = count
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        cache.append(frames)
    def sheet(slot):
        panel = Image.new('RGB', (5*264, 514), '#20242b')
        draw = ImageDraw.Draw(panel)
        for column, ((label, relative), frames) in enumerate(zip(inputs, cache)):
            im = frames[nearest[slot] if column == 0 else slot]
            if column == 2:
                im = im.crop((im.width//2, 0, im.width, im.height))
            im = ImageOps.contain(im, (264, 480))
            panel.paste(im, (column*264+(264-im.width)//2, 34+(480-im.height)//2))
            draw.text((column*264+3, 2), label, font=FONT, fill='white')
        draw.text((4, 19), f't={slot/25:.2f}s  model slot={slot}; native index={nearest[slot]}', font=FONT, fill='#ffda79')
        return panel
    contact = Image.new('RGB', (1320, 514*len(SLOTS)))
    for row, slot in enumerate(SLOTS):
        contact.paste(sheet(slot), (0, row*514))
    contact.save(OUT/f'{NAME}_stages.jpg', quality=94)
    with av.open(str(OUT/f'{NAME}_stages.mp4'), 'w') as writer:
        stream = writer.add_stream('libx264', rate=25)
        stream.width, stream.height, stream.pix_fmt = 1320, 514, 'yuv420p'
        stream.options = {'crf':'20', 'preset':'fast', 'threads':'2'}
        for slot in range(61):
            frame = av.VideoFrame.from_image(sheet(slot))
            for packet in stream.encode(frame): writer.mux(packet)
        for packet in stream.encode(): writer.mux(packet)
    projection = np.load(SRC/'motion/projection.npz')
    obs = projection['observed_coco17']
    with safe_open(SRC/'motion/motion.safetensors', framework='np') as f:
        pose = f.get_tensor('smpl_params_incam.body_pose').reshape(61, 21, 3)
    # SMPL-X joint 4/5 are knees; body_pose omits joint 0/root.
    knee = np.rad2deg(np.linalg.norm(pose[:, [3, 4]], axis=-1))
    rows = []
    for slot in range(61):
        rows.append(dict(slot=slot, seconds=slot/25, nearest_native_index=nearest[slot],
            left_knee_conf=float(obs[slot,13,2]), right_knee_conf=float(obs[slot,14,2]),
            left_ankle_conf=float(obs[slot,15,2]), right_ankle_conf=float(obs[slot,16,2]),
            left_knee_local_rotation_deg=float(knee[slot,0]), right_knee_local_rotation_deg=float(knee[slot,1])))
    with (OUT/f'{NAME}_motion.csv').open('w') as file:
        writer=csv.DictWriter(file, fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    report = dict(source=str(SRC), selected_slots=SLOTS, source_video_counts=counts, source_sha256=hashes,
        sampled_motion=[rows[s] for s in SLOTS],
        ankle_confidence_below_point3_frames=(obs[:,15:17,2]<.3).sum(0).tolist(),
        knee_rotation_max_deg=knee.max(0).tolist(), knee_rotation_max_slot=knee.argmax(0).tolist(),
        output_video_count=read_video(OUT/f'{NAME}_stages.mp4',set())[1],
        interpretation=['Local knee axis-angle rotation magnitude is a model output, not ground-truth flexion or anatomical joint angle.',
        'Detector confidence is not a calibrated occlusion probability.',
        'Native video shown at nearest actual exposure using frame_mapping brackets; model-stage outputs use identical slots.',
        'Only saved projection/model output inspected; no new model inference or optimized pose produced.'])
    (OUT/f'{NAME}_diagnostics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__': main()
