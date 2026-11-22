from pathlib import Path
import json,argparse
import numpy as np,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageOps,ImageFont
R=Path(__file__).resolve().parents[2]/'nonvehicle10';p=argparse.ArgumentParser();p.add_argument('--case',required=True);a=p.parse_args();cid=a.case;case=json.loads((R/'inputs'/cid/'segmented_manifest.json').read_text());out=R/'results'/cid;variants=['B0','R','H','B0_real1'];labels=['B0 | FLUX -> TRELLIS','R | real -> Recon','H | real + FLUX -> Recon','B0 + 1 real | TRELLIS'];font=ImageFont.truetype('/mnt/ojc/miniconda3/envs/4danyone/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf',17)
def tile(im,size):return ImageOps.pad(im.convert('RGB'),size,method=Image.Resampling.LANCZOS,color='#ddd')
src=next(r for r in case['images'] if r['id']==case['flux_reference_id']);flux=next(r for r in case['images'] if r['source_type']=='flux');source=Image.open(src['path']);gen=Image.open(flux['path']);sheet=Image.new('RGB',(1920,4*350),'white');d=ImageDraw.Draw(sheet)
for row,vi in enumerate([0,4,8,12]):
 for col,(im,label) in enumerate([(source,'Source (dataset crop)'),(gen,'FLUX (same across routes)')]+[(Image.open(out/f'{v}_seed42/orbit/view_{vi:02}.png'),labels[k]) for k,v in enumerate(variants)]):
  sheet.paste(tile(im,(320,310)),(col*320,row*350+38));d.text((col*320+5,row*350+2),label,font=font,fill='black');d.text((col*320+5,row*350+20),'Shared azimuth '+str(vi*22.5)+' deg' if col>=2 else ('Original retained' if col==0 else 'Generated evidence'),font=font,fill='black')
sheet.save(out/'comparison_orbit.jpg',quality=92)
readers=[imageio.get_reader(out/f'{v}_seed42/orbit/turntable.mp4') for v in variants];writer=imageio.get_writer(out/'comparison_video.mp4',fps=16,codec='libx264',quality=7,macro_block_size=2)
try:
 for i in range(96):
  canvas=Image.new('RGB',(1600,392),'#f4f4f4');d=ImageDraw.Draw(canvas);canvas.paste(tile(source,(160,320)),(0,38));canvas.paste(tile(gen,(160,320)),(160,38));d.text((4,9),'Source           FLUX',font=font,fill='black')
  for j,reader in enumerate(readers):canvas.paste(tile(Image.fromarray(reader.get_data(i)),(320,320)),((j+1)*320,38));d.text(((j+1)*320+5,9),labels[j],font=font,fill='black')
  d.text((5,369),cid+' | normalized units; backend yaw preserved; no matched-source view alignment',font=font,fill='black');writer.append_data(np.array(canvas))
finally:
 writer.close()
 for reader in readers:reader.close()
print('COMPOSED',cid,flush=True)
