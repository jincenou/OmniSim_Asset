"""Display actual sampled alpha and initialized geometry; no rendered asset claim."""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from bridge_export import project

ROOT = Path(__file__).resolve().parents[1]
LABEL = os.environ.get('FDANYONE_BRIDGE_LABEL', 'official_sample')
assert LABEL in ('official_sample', 'waymo_sample')
SOURCE = Path(os.environ.get('FDANYONE_BRIDGE_SOURCE', str(ROOT.parent/'outputs/turbo')))


def main():
    dest = ROOT/'bridge'/LABEL
    cams = json.loads((SOURCE/'cameras.json').read_text())['cameras']
    init = np.load(dest/'init.npz')
    canvas = Image.new('RGB', (176*6, 344*3), 'white')
    draw = ImageDraw.Draw(canvas)
    y, x = np.indices((320, 176))
    checker = np.where(((x//12+y//12) % 2)[..., None], 224, 255)
    for frame in range(3):
        points = init['positions'][np.isclose(init['times'], frame/2)]
        for col, view in enumerate((0, 4, 8, 12, 16, 20)):
            rgba = np.asarray(Image.open(dest/'images'/f'cam{view:02d}'/f'{frame:06d}.png'))
            alpha = rgba[..., 3:]/255.
            rgb = np.rint(rgba[..., :3]*alpha+checker*(1-alpha)).astype(np.uint8)
            uv, valid = project(points, cams[view])
            rgb[uv[valid, 1], uv[valid, 0]] = [230, 40, 40]
            canvas.paste(Image.fromarray(rgb), (176*col, 344*frame+24))
            draw.text((176*col+3, 344*frame+5), f'V{view:02d} t={frame*2.4:.1f}s | hull', fill='black')
    output = ROOT/'previews'/f'{LABEL}_bridge.jpg'
    canvas.save(output, quality=92)
    print(output, flush=True)


if __name__ == '__main__':
    main()
