import av,json
from pathlib import Path
from PIL import Image,ImageDraw
W=Path(__file__).resolve().parents[1];out=W/'sources/bmw/seek_near';out.mkdir(parents=True,exist_ok=True);rows=[]
with av.open('https://video.golem.de/download/26545?q=medium',timeout=(20,40)) as c:
 st=c.streams.video[0]
 for t in [12,16,20,24,28,32,36,40,44]:
  c.seek(int(t/st.time_base),stream=st,backward=True)
  for f in c.decode(st):
   if float(f.pts*f.time_base)<t:continue
   p=out/f'{t:03}.png';f.to_image().save(p);rows.append(dict(image_path=str(p),pts=f.pts,time_base=str(f.time_base),source_time_s=float(f.pts*f.time_base)));print(t,flush=True);break
  (out/'frames.json').write_text(json.dumps(rows,indent=2))
sheet=Image.new('RGB',(960,615),'white');draw=ImageDraw.Draw(sheet)
for i,r in enumerate(rows):
 im=Image.open(r['image_path']);im.thumbnail((320,180));x=i%3*320;y=i//3*205;sheet.paste(im,(x,y));draw.text((x+4,y+182),str(r['source_time_s']),fill='black')
sheet.save(out/'sheet.jpg')
