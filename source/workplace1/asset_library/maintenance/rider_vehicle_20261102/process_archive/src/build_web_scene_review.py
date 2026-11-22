from pathlib import Path
import json
from PIL import Image,ImageDraw
W=Path(__file__).resolve().parents[1];names=[r['name'] for r in json.loads((W/'sources/web_expansion_20261103/queue.json').read_text())];ready=[]
for n in names:
 r=W/'runs'/n;p=r/'insertion_near_follow';flag=r/'insert_near_follow.exit'
 if flag.exists() and flag.read_text().strip()=='0' and all((p/'frames'/f'f{i:03d}.png').exists() for i in [0,12,23]):ready.append(n)
for start in range(0,len(ready),3):
 group=ready[start:start+3];sheet=Image.new('RGB',(1140,len(group)*340),'white');d=ImageDraw.Draw(sheet)
 for j,n in enumerate(group):
  for i,f in enumerate([0,12,23]):
   im=Image.open(W/'runs'/n/'insertion_near_follow/frames'/f'f{f:03d}.png').crop((1040,120,1420,440));sheet.paste(im,(i*380,j*340+20));d.text((i*380+3,j*340+3),n+' frame '+str(f),fill='black')
 sheet.save(W/'reports'/f'web_scene_review_{start:02d}.jpg')
print('COMPLETE_FOR_REVIEW',ready)
