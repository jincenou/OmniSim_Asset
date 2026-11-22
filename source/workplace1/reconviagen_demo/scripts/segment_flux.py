"""Use the existing DriveLab BiRefNet-lite remover on archived FLUX RGB."""
from pathlib import Path
import os,json,argparse,hashlib
ROOT=Path(__file__).resolve().parents[1]
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'cache/numba')
os.environ['U2NET_HOME']=str(ROOT/'cache/u2net')
os.environ['OMP_NUM_THREADS']='4'
cache=ROOT/'cache/u2net';cache.mkdir(exist_ok=True)
weight=cache/'birefnet-general-lite.onnx'
if not weight.exists():weight.symlink_to('/mnt/wmj/workspace4/pipeline/birefnet-general-lite.onnx')
from PIL import Image,ImageDraw,ImageOps
import numpy as np
import rembg
p=argparse.ArgumentParser();p.add_argument('--case',required=True);a=p.parse_args()
out=ROOT/'inputs'/a.case;case=json.loads((out/'segmented_manifest.json').read_text())
session=rembg.new_session('birefnet-general-lite',providers=['CPUExecutionProvider'])
for rec in case['images']:
    if rec['source_type']!='flux':continue
    image=Image.open(rec['path']).convert('RGB');rgba=rembg.remove(image,session=session)
    alpha=rgba.getchannel('A');mask=np.asarray(alpha)
    if (mask>204).sum()<100:raise ValueError('Empty FLUX foreground')
    rgba.save(rec['rgba_path']);alpha.save(rec['mask_path'])
    rec.update(mask_model='DriveLab original birefnet-general-lite.onnx',mask_foreground_fraction=float((mask>204).mean()),
               mask_method='learned background removal; no white-pixel deletion')
    rec.pop('mask_prompt_box',None);rec.pop('mask_score',None)
(out/'segmented_manifest.json').write_text(json.dumps(case,indent=2))
tiles=[]
for rec in case['images']:
    rgba=Image.open(rec['rgba_path']);preview=Image.new('RGBA',rgba.size,(80,80,80,255));preview.alpha_composite(rgba)
    tile=Image.new('RGB',(320,290),'#ddd');im=ImageOps.contain(preview.convert('RGB'),(320,250));tile.paste(im,(0,25))
    ImageDraw.Draw(tile).text((3,3),f"{rec['id']} / {rec['split']}",fill='black');tiles.append(tile)
sheet=Image.new('RGB',(1280,290*((len(tiles)+3)//4)),'#ddd')
for i,tile in enumerate(tiles):sheet.paste(tile,(i%4*320,i//4*290))
sheet.save(out/'segmented_inputs.jpg')
print(a.case,'FLUX masks fixed with original remover',flush=True)
