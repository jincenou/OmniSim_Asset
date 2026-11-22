import json
from pathlib import Path
import imageio.v2 as io
import numpy as np
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;rs=json.loads((B.parent/'v2/cycle_updates.json').read_text())['assets'];rs=[r for r in rs if r.get('error','').startswith('unclassified')]
(B/'order.json').write_text(json.dumps([r['folder'] for r in rs]))
for group in range(10):
 canvas=Image.new('RGB',(1440,5*245),'#eeeeee');draw=ImageDraw.Draw(canvas)
 for j,r in enumerate(rs[group*5:group*5+5]):
  reader=io.get_reader('/mnt/ojc/workplace1/asset_library/human/previews/'+r['folder']+'.mp4');frames=[f for f in reader];reader.close();indices=np.linspace(0,len(frames)-1,6,dtype=int)
  for k,idx in enumerate(indices):
   im=Image.fromarray(frames[idx]);w,h=im.size;im=im.crop((int(w*.6),32,w,h)).resize((240,220));canvas.paste(im,(k*240,j*245+25));draw.text((k*240+4,j*245+5),r['folder']+' frame '+str(idx),fill='black')
 canvas.save(B/f'sheet_{group:02d}.jpg',quality=95)
 print(group,flush=True)
