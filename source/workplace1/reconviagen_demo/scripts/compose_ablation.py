"""Compare actual pickup count/seed renders under verified shared cameras."""
from pathlib import Path
import json,argparse
import numpy as np
from PIL import Image,ImageDraw
import imageio.v2 as imageio
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--case',default='fresh_black_pickup');a=p.parse_args()
out=ROOT/'results'/a.case
tags=['R1_seed42','R2_seed42','R_seed42','R_seed7']
assets=[out/t/'asset' for t in tags]
configs=[json.loads((a/'render_config.json').read_text()) for a in assets]
for key in ['length_m','distance_m','size','focal_px','video_frames','video_fps','center_m']:
 if len({str(c[key]) for c in configs})!=1:raise ValueError(key)
if not all(json.loads((a/'experiment.json').read_text())['heading_verified'] for a in assets):raise ValueError('Unreviewed heading')
labels=['R1 / seed 42','R2 / seed 42','R4 / seed 42','R4 / seed 7'];readers=[imageio.get_reader(a/'turntable.mp4') for a in assets]
writer=imageio.get_writer(out/'ablation_count_seed.mp4',fps=16,codec='libx264',quality=8)
try:
 for i in range(96):
  im=Image.new('RGB',(1024,1088),'#ddd');dr=ImageDraw.Draw(im)
  for j,r in enumerate(readers):
   x=j%2*512;y=j//2*544;im.paste(Image.fromarray(r.get_data(i)),(x,y+32));dr.text((x+8,y+8),labels[j],fill='black')
  writer.append_data(np.asarray(im))
  if i==6:im.save(out/'ablation_count_seed.jpg')
finally:
 writer.close()
 for r in readers:r.close()
(out/'ablation_count_seed.json').write_text(json.dumps(dict(tags=tags,real_count=[1,2,4,4],seeds=[42,42,42,7],eight_views='Not run: additional candidates severely occluded or near duplicate',same_camera=True,heading='+X',heldout_used=False),indent=2))
