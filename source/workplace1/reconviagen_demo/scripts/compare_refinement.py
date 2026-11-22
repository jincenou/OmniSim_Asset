"""Compare each refined asset to the base from the exact same invocation."""
from pathlib import Path
import json,argparse
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--case',default='fresh_black_pickup');a=p.parse_args()
out=ROOT/'results'/a.case
tags=['R_refine_base','R_refined','F_refine_base','F_refined']
assets=[out/f'{tag}_seed42/asset' for tag in tags]
configs=[json.loads((a/'render_config.json').read_text()) for a in assets]
for key in ['length_m','distance_m','size','focal_px','video_frames','video_fps','center_m']:
    assert len({str(c[key]) for c in configs})==1,key
assert all(json.loads((a/'experiment.json').read_text())['heading_verified'] for a in assets)
labels=['R: same-run base','R: official refinement','F: same-run base','F: official refinement']
readers=[imageio.get_reader(a/'turntable.mp4') for a in assets]
writer=imageio.get_writer(out/'refinement_R_F_comparison.mp4',fps=16,codec='libx264',quality=8)
try:
    for i in range(96):
        canvas=Image.new('RGB',(1024,1088),'#ddd');draw=ImageDraw.Draw(canvas)
        for j,reader in enumerate(readers):
            x=j%2*512;y=j//2*544
            canvas.paste(Image.fromarray(reader.get_data(i)),(x,y+32));draw.text((x+8,y+8),labels[j],fill='black')
        writer.append_data(np.asarray(canvas))
        if i==6:canvas.save(out/'refinement_R_F_comparison.jpg')
finally:
    writer.close()
    for reader in readers:reader.close()
(out/'refinement_R_F_comparison.json').write_text(json.dumps(dict(tags=tags,seed=42,
    before_source='base_repeat.ply from each successful refine invocation',
    same_camera=True,heldout_used=False,scope='Visual before/after; no held-out accuracy claim'),indent=2))
print(out/'refinement_R_F_comparison.mp4')
