"""Decode a bounded preview with exact decoder PTS, preserving sampled frame lineage."""
import sys,json,hashlib
from pathlib import Path
import av
from PIL import Image,ImageDraw
p=Path(sys.argv[1]);out=p.parent/(p.stem+'_frames');out.mkdir(exist_ok=True)
step=float(sys.argv[2]) if len(sys.argv)>2 else 1.
rows=[];target=0.
with av.open(str(p)) as c:
 stream=c.streams.video[0]
 for i,f in enumerate(c.decode(stream)):
  if f.pts is None:raise ValueError('Missing original presentation timestamp')
  t=float(f.pts*f.time_base)
  if t+1e-6<target:continue
  im=f.to_image();im.save(out/f'{len(rows):03}.jpg');rows.append(dict(processed_frame_index=len(rows),native_frame_index=i,pts=f.pts,time_base=str(f.time_base),source_time_s=t,image_path=str(out/f'{len(rows):03}.jpg')));target=t+step
(out/'frames_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
for start in range(0,len(rows),30):
 subset=rows[start:start+30];sheet=Image.new('RGB',(5*320,((len(subset)+4)//5)*205),'#eee');draw=ImageDraw.Draw(sheet)
 for j,r in enumerate(subset):
  im=Image.open(r['image_path']);im.thumbnail((320,180));x=j%5*320;y=j//5*205;sheet.paste(im,(x,y));draw.text((x+4,y+183),f"{r['processed_frame_index']}  {r['source_time_s']:.3f}s",fill='black')
 sheet.save(out/f'sheet_{start//30}.jpg')
print(out,len(rows))
