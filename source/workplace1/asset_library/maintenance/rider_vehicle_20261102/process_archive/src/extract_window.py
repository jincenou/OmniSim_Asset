import av,sys,json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
src,lo,hi,dest=sys.argv[1:5];lo=float(lo);hi=float(hi);out=Path(dest);out.mkdir(parents=True,exist_ok=True);rows=[];target=lo
with av.open(src,timeout=(20,40)) as c:
 st=c.streams.video[0];c.seek(int(lo/st.time_base),stream=st,backward=True)
 for f in c.decode(st):
  t=float(f.pts*f.time_base)
  if t<target:continue
  if t>hi:break
  p=out/f'{len(rows):03}.png';f.to_image().save(p);rows.append(dict(image_path=str(p),pts=f.pts,time_base=str(f.time_base),source_time_s=t,source=src,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),native_wh=[f.width,f.height]));target=t+.18
(out/'frames.json').write_text(json.dumps(rows,indent=2))
sheet=Image.new('RGB',(1280,205*((len(rows)+3)//4)),'white');draw=ImageDraw.Draw(sheet)
for i,r in enumerate(rows):
 im=Image.open(r['image_path']);im.thumbnail((320,180));x=i%4*320;y=i//4*205;sheet.paste(im,(x,y));draw.text((x+4,y+182),f"{i} {r['source_time_s']:.3f}s",fill='black')
sheet.save(out/'sheet.jpg');print(out,len(rows))
