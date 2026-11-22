"""Deterministic video sample sheets; no enhancement or generated replacement images."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys

R = Path(__file__).resolve().parents[1]


def render(root):
    import av
    from PIL import Image, ImageDraw
    times = tuple(range(0, 61, 10))
    videos = sorted((root / 'videos').glob('*.mp4'))
    assert len(videos) == 6
    tile_w, tile_h = 176, 320
    sheet = Image.new('RGB', (tile_w * 6, (tile_h + 20) * len(times)), 'white')
    draw = ImageDraw.Draw(sheet)
    evidence = []
    for view, path in enumerate(videos):
        selected = set()
        with av.open(str(path)) as container:
            count = 0
            for index, frame in enumerate(container.decode(video=0)):
                count += 1
                if index in times:
                    row = times.index(index)
                    sheet.paste(frame.to_image().resize((tile_w, tile_h)), (view * tile_w, row * (tile_h + 20) + 20))
                    draw.text((view * tile_w + 3, row * (tile_h + 20) + 3), f'view {view}, frame {index}', fill='black')
                    selected.add(index)
        assert count == 61 and selected == set(times)
        evidence.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'decoded_frames': count})
    out = root / 'short_preview'
    out.mkdir(exist_ok=True)
    image = out / 'dense_contact.jpg'
    if image.exists():
        raise FileExistsError(image)
    sheet.save(image, quality=94)
    (out / 'dense_contact_provenance.json').write_text(json.dumps({'frames': times, 'views': 6, 'sources': evidence,
        'image': str(image), 'sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
        'operation': 'resize for display only; original video pixels, no enhancement'}, indent=2) + '\n')


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['M02'], required=True)
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        render(args.worker)
        return
    async def run(variant):
        root = R / f'experiments/moving_{args.case}_short61_{variant}_v1/outputs/smoke'
        child = await asyncio.create_subprocess_exec(sys.executable, __file__, '--case', args.case, '--worker', str(root))
        code = await child.wait()
        if code:
            raise RuntimeError(f'{variant} contact generation exited {code}')
    await asyncio.gather(*map(run, ('M0', 'M2')))


if __name__ == '__main__':
    asyncio.run(main())
