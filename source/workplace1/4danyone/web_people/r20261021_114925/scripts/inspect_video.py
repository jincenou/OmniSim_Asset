"""Decode the entire original video; preserve real presentation timestamps."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import av
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
PROBE = '/mnt/ojc/miniconda3/envs/qwen35_vlm/bin/ffprobe'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('id')
    args = parser.parse_args()
    path = ROOT/'raw'/f'{args.id}.webm'
    meta = json.loads(subprocess.check_output([PROBE, '-v', 'error', '-show_streams',
                                              '-show_format', '-of', 'json', str(path)]))
    (ROOT/f'sources/{args.id}_ffprobe.json').write_text(json.dumps(meta,indent=2))
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 2
        frames = [(float(f.pts*f.time_base), f.to_image()) for f in container.decode(stream)]
    times = np.array([t for t,_ in frames])
    assert len(times)>121 and np.isfinite(times).all() and (np.diff(times)>0).all()
    indices = np.unique(np.rint(np.linspace(0,len(frames)-1,16)).astype(int))
    sheet = Image.new('RGB',(1280,4*205),'#222222')
    draw = ImageDraw.Draw(sheet)
    for k,i in enumerate(indices):
        x,y=k%4*320,k//4*205
        sheet.paste(ImageOps.pad(frames[i][1],(320,180),color='#222222'),(x,y+25))
        draw.text((x+4,y+4),f'{args.id} frame {i} t={times[i]:.3f}s',fill='white')
    sheet.save(ROOT/f'sources/{args.id}_contact.jpg',quality=93)
    record = dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  decoded_frames=len(frames),timestamps_s=times.tolist(),
                  contact_frame_indices=indices.tolist(),full_decode=True,visual_review=False)
    (ROOT/f'sources/{args.id}_decode.json').write_text(json.dumps(record,indent=2))
    print(args.id,len(frames),times[-1],flush=True)


if __name__ == '__main__':
    main()
