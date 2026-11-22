from pathlib import Path
import os,json,time
os.environ['U2NET_HOME']='/mnt/wmj/workspace4/pipeline'
import rembg
from PIL import Image,ImageOps,ImageDraw
W=Path(__file__).resolve().parents[1];work=W/'runs/web_expansion_20261103_flux';session=rembg.new_session('birefnet-general-lite',providers=['CPUExecutionProvider']);outputs=[]
for i in range(7):
 p=work/f'asset_{i:05d}/flux_0.png'
 for _ in range(180):
  state=json.loads((work/'generation.json').read_text())['assets'][i]
  if state['stages'].get('flux',{}).get('status')=='succeeded':break
  if state['stages'].get('flux',{}).get('status')=='failed':raise RuntimeError(state)
  time.sleep(2)
 else:raise TimeoutError(str(p))
 im=Image.open(p).convert('RGB');rgba=rembg.remove(im,session=session);rgba.save(p.with_name('flux_rgba.png'));white=Image.new('RGB',im.size,'white');white.paste(rgba,mask=rgba.getchannel('A'));white.save(p.with_name('flux_matte_preview.jpg'));outputs.append(p.with_name('flux_matte_preview.jpg'));print('MATTE',i,flush=True)
sheet=Image.new('RGB',(1400,900),'white');d=ImageDraw.Draw(sheet)
for i,p in enumerate(outputs):
 im=ImageOps.contain(Image.open(p),(345,420));x=i%4*350;y=i//4*450;sheet.paste(im,(x,y+25));d.text((x+3,y+3),str(i),fill='black')
sheet.save(work/'matte_sheet.jpg')
