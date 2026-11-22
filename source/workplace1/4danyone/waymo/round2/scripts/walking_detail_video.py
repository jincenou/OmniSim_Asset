"""CPU display crops of the existing synchronized, unfiltered comparison."""
import json
import hashlib
from pathlib import Path
import av
import numpy as np
from PIL import Image, ImageDraw, ImageOps

R = Path(__file__).resolve().parents[1]
OUT = R / 'diagnostics/walking_temporal'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = R / 'comparisons/walk063_rootlocal24_validation/view08.mp4'
    target = OUT / 'motion_details.mp4'
    if target.exists():
        raise FileExistsError(target)
    # A single union box over time avoids introducing crop tracking motion.
    data = R / 'reconstruction/data/walk063_rootlocal24/images/cam08'
    bounds = []
    for path in sorted(data.glob('*.png')):
        with Image.open(path) as im:
            ys, xs = np.where(np.asarray(im)[..., 3] > 127)
        if not len(xs):
            raise ValueError('Empty foreground')
        bounds.append((xs.min(), ys.min(), xs.max()+1, ys.max()+1))
    assert len(bounds) == 121
    b = np.array(bounds)
    x0, y0 = b[:, :2].min(0)
    x1, y1 = b[:, 2:].max(0)
    h = y1-y0
    regions = {
        'Head / hair': (max(0, int(x0)-8), max(0, int(y0)-8), min(352, int(x1)+8), int(y0+.27*h)),
        'Hands / torso': (max(0, int(x0)-8), int(y0+.2*h), min(352, int(x1)+8), int(y0+.74*h)),
        'Feet / lower legs': (max(0, int(x0)-8), int(y0+.72*h), min(352, int(x1)+8), min(640, int(y1)+8)),
    }
    labels = ['Generated', 'Original 5k', 'Original 50k', 'Dense init 50k']
    sampled = [0, 29, 50, 55, 60, 74, 98, 120]
    sheets = []
    with av.open(str(source)) as src, av.open(str(target), 'w') as dst:
        src.streams.video[0].thread_count = 1
        out = dst.add_stream('libx264', rate=25)
        out.width, out.height, out.pix_fmt = 1280, 768, 'yuv420p'
        out.options = {'crf': '18', 'preset': 'fast', 'threads': '2'}
        count = 0
        for i, frame in enumerate(src.decode(video=0)):
            assert frame.width == 1408 and frame.height == 664
            assert abs(float(frame.pts*frame.time_base)-i/25) < 1e-7
            original = frame.to_image()
            sheet = Image.new('RGB', (1280, 768), '#333333')
            draw = ImageDraw.Draw(sheet)
            for row, (region, box) in enumerate(regions.items()):
                for col, label in enumerate(labels):
                    # The existing comparison has a 24px title above each image.
                    bx = (box[0]+col*352, box[1]+24, box[2]+col*352, box[3]+24)
                    crop = ImageOps.contain(original.crop(bx), (312, 226), Image.Resampling.NEAREST)
                    sheet.paste(crop, (col*320+(320-crop.width)//2, row*256+26))
                    draw.text((col*320+4, row*256+4), f'{label} | {region}', fill='white')
            draw.text((4, 756), f'f{i:03d} / 120 | t={i/25:.2f}s | view08 | display crop, no enhancement', fill='white')
            if i in sampled:
                sheet.save(OUT/f'detail_frame{i:03d}.jpg', quality=95)
                sheets.append(sheet.resize((640, 384)))
            for packet in out.encode(av.VideoFrame.from_image(sheet)):
                dst.mux(packet)
            count += 1
        for packet in out.encode():
            dst.mux(packet)
    assert count == 121
    with av.open(str(target)) as check:
        decoded = list(check.decode(video=0))
        assert len(decoded) == 121
        assert all(f.width == 1280 and f.height == 768 and abs(float(f.pts*f.time_base)-i/25) < 1e-7 for i, f in enumerate(decoded))
    contact = Image.new('RGB', (1280, 384*4))
    for i, sheet in enumerate(sheets):
        contact.paste(sheet, ((i%2)*640, (i//2)*384))
    contact.save(OUT/'details_contact.jpg', quality=95)
    report = {'state': 'encoded_decoded_visual_review_pending', 'source': str(source), 'source_sha256': sha(source),
              'output_sha256': sha(target), 'frames': count, 'fps': 25, 'view': 8, 'regions': regions,
              'sampled_frames': sampled, 'runs': labels, 'camera_time_identical': True,
              'scope': 'Display crops from existing compressed comparison video, not metric inputs. Nearest pixel enlargement; no enhancement, no filtering. Full original comparison retains outside-frame content.'}
    (OUT/'detail_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
