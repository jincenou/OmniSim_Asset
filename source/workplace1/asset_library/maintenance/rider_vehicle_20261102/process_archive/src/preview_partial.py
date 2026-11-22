import av,sys,json
from pathlib import Path
from PIL import Image,ImageDraw
p=Path(sys.argv[1]);out=p.parent/'partial_preview';out.mkdir(exist_ok=True);rows=[];target=0
try:
 with av.open(str(p)) as c:
  st=c.streams.video[0];print(p.parent.name,'STREAM',st.width,st.height,'duration',c.duration,flush=True)
  for i,f in enumerate(c.decode(st)):
   t=float(f.pts*f.time_base)
   if t<target:continue
   path=out/f'{len(rows):03}.jpg';f.to_image().save(path);rows.append(dict(index=i,pts=f.pts,time_base=str(f.time_base),seconds=t,path=str(path)));target=t+2
except Exception as e:print('PARTIAL_END',type(e).__name__,str(e)[:120],flush=True)
(out/'frames.json').write_text(json.dumps(rows,indent=2))
for off in range(0,len(rows),30):
 sub=rows[off:off+30];sheet=Image.new('RGB',(1600,205*((len(sub)+4)//5)),'white');draw=ImageDraw.Draw(sheet)
 for j,r in enumerate(sub):
  im=Image.open(r['path']);im.thumbnail((320,180));x=j%5*320;y=j//5*205;sheet.paste(im,(x,y));draw.text((x+4,y+182),f"{off+j} {r['seconds']:.3f}s",fill='black')
 sheet.save(out/f'sheet_{off//30}.jpg')
print('PREVIEW',len(rows),flush=True)
