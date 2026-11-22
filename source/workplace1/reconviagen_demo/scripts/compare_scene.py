"""Compare decoded original SplatAD outputs, plus a front-camera detail view."""
from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image,ImageDraw
import imageio.v2 as imageio
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--scene-dir',default='scene0104');a=p.parse_args()
out=ROOT/'results'/a.case;dirs=[out/f'{v}_seed42'/a.scene_dir for v in ['B0','R']]
metas=[json.loads((d/'meta.json').read_text()) for d in dirs]
for k in ['config','traj_npz','step','n_frames','start_frame','target_length_m']:
 if metas[0][k]!=metas[1][k]:raise ValueError(k)
for m in metas:
 assert all(not x['missing'] and not x['unexpected'] for x in m['checkpoint_load_audit'])
readers=[imageio.get_reader(d/'insert_splatad_6cam.mp4') for d in dirs]
label=['B0: original TRELLIS / FLUX pair','R: ReconViaGen / real views']
dest=out/f'{a.scene_dir}_comparison_B0_R.mp4';writer=imageio.get_writer(dest,fps=12,codec='libx264',quality=8)
try:
 for i in range(metas[0]['n_frames']):
  canvas=Image.new('RGB',(1600,640),'#ddd');dr=ImageDraw.Draw(canvas)
  for j,r in enumerate(readers):
   frame=Image.fromarray(r.get_data(i));w,h=frame.size
   # Preserve the complete six-camera grid; enlarge front-left/front below.
   canvas.paste(frame.resize((800,300)),(j*800,25))
   detail=frame.crop((0,0,w*2//3,h//2));detail.thumbnail((800,300));canvas.paste(detail,(j*800,340))
   dr.text((j*800+8,7),label[j],fill='black');dr.text((j*800+8,325),'Front-left + front detail / same frame',fill='black')
  writer.append_data(np.asarray(canvas))
  if i==30:canvas.save(out/f'{a.scene_dir}_comparison_B0_R.jpg')
finally:
 writer.close()
 for r in readers:r.close()
(out/f'{a.scene_dir}_comparison_B0_R.json').write_text(json.dumps(dict(inputs=[str(d) for d in dirs],shared_settings=metas[0],scope='Actual SplatAD insertion; visual contact quality reviewed separately'),indent=2))
print(dest)
