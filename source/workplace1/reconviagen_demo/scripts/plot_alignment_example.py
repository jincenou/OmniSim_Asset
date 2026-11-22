from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageOps
from evaluate_asset_metrics import foreground
from evaluate_pixel_metrics import native_square
R=Path(__file__).resolve().parents[1];c='random10_04_white_suv';cam=json.loads((R/'inputs'/c/'evaluation/camera.json').read_text());vs=['B0','R','F','H','B0_real1','B0_real4'];sheet=Image.new('RGB',(1792,650),'white');d=ImageDraw.Draw(sheet)
for row,(mode,func) in enumerate([('A: foreground normalized',foreground),('B: camera projection preserved',native_square)]):
 y=row*325;d.rectangle((0,y,1792,y+32),fill='#e2e8f0');d.text((10,y+10),mode,fill='black')
 pairs=[('Source',cam['source_rgb'],cam['source_mask'])]+[(v,R/'results'/c/f'{v}_seed42/evaluation/matched_source.png',R/'results'/c/f'{v}_seed42/evaluation/matched_source_alpha.png') for v in vs]
 for i,(name,rgb,mask) in enumerate(pairs):
  im,*_=func(rgb,mask);sheet.paste(im.resize((256,256),Image.Resampling.LANCZOS),(i*256,y+65));d.text((i*256+8,y+43),name,fill='black')
sheet.save(R/'deliveries/vehicle_experiments_20261023/images/overview/alignment_example.jpg',quality=93)
