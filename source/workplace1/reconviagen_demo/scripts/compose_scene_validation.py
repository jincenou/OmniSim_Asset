from pathlib import Path
import json,argparse
import numpy as np,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageOps,ImageFont
R=Path(__file__).resolve().parents[1];O=R/'scene_validation';p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--case',required=True);a=p.parse_args();base=O/'results'/a.scene/a.case;variants=['B0','R','H','B0_real4'];font=ImageFont.truetype('/mnt/ojc/miniconda3/envs/4danyone/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf',21);metas=[json.loads((base/v/'meta.json').read_text()) for v in variants];assert len({m['trajectory_sha256'] for m in metas})==1
# FRONT is tile (row0,col1); use a common fixed crop containing all four routes over all60frames.
boxes=[]
for v in variants:
 for row in json.loads((base/v/'visibility_diagnostics.json').read_text()):
  if row['sensor']=='FRONT' and row.get('projected_bbox'):boxes.append(row['projected_bbox'])
if not boxes:raise ValueError('No FRONT evidence')
b=np.array(boxes);lo=b[:,:2].min(0)/2;hi=b[:,2:].max(0)/2;center=(lo+hi)/2;size=np.maximum((hi-lo)*1.65,[400,225]);size[0]=max(size[0],size[1]*16/9);size[1]=size[0]*9/16;size=np.minimum(size,[800,450]);x0=int(np.clip(center[0]-size[0]/2,0,800-size[0]));y0=int(np.clip(center[1]-size[1]/2,0,450-size[1]));crop=[x0,y0,int(x0+size[0]),int(y0+size[1])]
(base/'comparison_crop.json').write_text(json.dumps(dict(camera='FRONT',shared_fixed_crop_in_downscaled_camera=crop,selection='union of all routes/all60frames +65percent context, min400x225; no per-route or per-frame crop fitting'),indent=2))
def panel(path):
 im=Image.open(path).crop((800,0,1600,450)).crop(crop);return im.resize((720,405),Image.Resampling.LANCZOS)
writer=imageio.get_writer(base/'comparison_scene.mp4',fps=12,codec='libx264',quality=7,macro_block_size=2)
try:
 for i in range(60):
  canvas=Image.new('RGB',(1440,912),'#f4f4f4');d=ImageDraw.Draw(canvas);d.text((10,6),a.scene+' | '+a.case+' | shared trajectory / scale rule / camera / road fit',font=font,fill='black')
  for j,v in enumerate(variants):
   x=j%2*720;y=j//2*440+40;canvas.paste(panel(base/v/'frames'/f'f{i:03}.png'),(x,y+30));d.text((x+8,y),v,font=font,fill='black')
  writer.append_data(np.array(canvas))
  if i in [0,30,59]:canvas.save(base/f'comparison_frame_{i:03}.jpg',quality=92)
finally:writer.close()
print('COMPOSED',a.scene,a.case,flush=True)
