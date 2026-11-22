"""Build aligned four-route posters and videos from actual rendered assets."""
from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image,ImageDraw
import imageio.v2 as imageio
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--case',required=True);a=p.parse_args()
variants=['B0','R','F','H'];out=ROOT/'results'/a.case;assets=[out/f'{v}_seed42/asset' for v in variants]
configs=[json.loads((x/'render_config.json').read_text()) for x in assets]
keys=['length_m','distance_m','focal_px','size','elevation_deg','center_m','background','azimuths_deg','video_frames','video_fps']
for c in configs[1:]:
    if any(c[k]!=configs[0][k] for k in keys):raise ValueError('Comparison cameras differ')
labels=['B0 | FLUX 2 / original TRELLIS','R | real views / ReconViaGen','F | FLUX 2 / ReconViaGen','H | real + FLUX / ReconViaGen']
size=configs[0]['size'];pad=32
poster=Image.new('RGB',(size*4,(size+pad)*3),'#eeeeee');draw=ImageDraw.Draw(poster)
for col,(asset,label) in enumerate(zip(assets,labels)):
    for row,view in enumerate([1,4,8]):
        x=col*size;y=row*(size+pad);poster.paste(Image.open(asset/f'view_{view:02}.png'),(x,y+pad))
        draw.text((x+8,y+8),f'{label} | {view*22.5:g} deg',fill='black')
poster.save(out/'comparison_B0_R_F_H.jpg',quality=94)
readers=[imageio.get_reader(x/'turntable.mp4') for x in assets]
writer=imageio.get_writer(out/'comparison_B0_R_F_H.mp4',fps=configs[0]['video_fps'],codec='libx264',quality=8)
try:
    for i in range(configs[0]['video_frames']):
        grid=Image.new('RGB',(size*2,(size+pad)*2),'#eeeeee');d=ImageDraw.Draw(grid)
        for j,(reader,label) in enumerate(zip(readers,labels)):
            x=j%2*size;y=j//2*(size+pad);grid.paste(Image.fromarray(reader.get_data(i)),(x,y+pad));d.text((x+8,y+8),label,fill='black')
        writer.append_data(np.asarray(grid))
finally:
    writer.close()
    for reader in readers:reader.close()
(out/'comparison_config.json').write_text(json.dumps(dict(variants=variants,render_config=configs[0],heading='all reviewed front +X',
    note='Different input information and identity drift; not a claim of algorithmic superiority'),indent=2))
print(out/'comparison_B0_R_F_H.mp4')
