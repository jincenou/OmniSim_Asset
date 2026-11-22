"""Export every model frame for inspection; exact sampled source and alpha."""
from pathlib import Path
import sys,json
import av,numpy as np
from PIL import Image,ImageDraw,ImageOps
from prepare_person import ROOT

for aid in sys.argv[1:]:
 d=ROOT/'inputs'/aid;dest=ROOT/'reviews'/aid;dest.mkdir(exist_ok=True)
 mapping=json.loads((d/'inputs/frame_mapping.json').read_text())
 masks=np.load(d/'segmentation/target_masks.npz')['masks'];tiles=[]
 with av.open(str(d/'crop_unsegmented.mp4')) as c:
  for i,f in enumerate(c.decode(video=0)):
   rgb=f.to_ndarray(format='rgb24');alpha=masks[i].astype(np.float32)[...,None]/255
   im=Image.fromarray(np.rint(rgb*alpha+255*(1-alpha)).clip(0,255).astype('uint8'))
   tile=Image.new('RGB',(192,360),'white')
   for col,pic in enumerate([f.to_image(),im]):
    pic=ImageOps.contain(pic,(96,330));tile.paste(pic,(col*96+(96-pic.width)//2,20))
   ImageDraw.Draw(tile).text((2,2),f'{i} / {i/25:.2f}s',fill='black');tiles.append(tile)
 assert len(tiles)==121
 for start in range(0,len(tiles),32):
  subset=tiles[start:start+32];sheet=Image.new('RGB',(192*8,360*((len(subset)+7)//8)),'#dddddd')
  for k,tile in enumerate(subset):sheet.paste(tile,((k%8)*192,(k//8)*360))
  sheet.save(dest/f'all_frames_{start:03d}.jpg',quality=92)
 (dest/'inspection_scope.json').write_text(json.dumps(dict(frames_exported=121,review_passed=False,method='all consecutive model-frame RGB/alpha pairs; exported contact sheets are inspection evidence, not automatic visual acceptance'),indent=2))
 print(aid,'121 frames exported',flush=True)
