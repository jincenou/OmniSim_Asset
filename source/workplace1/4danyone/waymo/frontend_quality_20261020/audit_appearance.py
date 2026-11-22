"""Small visual survey. No auto quality acceptance based on image darkness."""
from pathlib import Path
import json
import av
from PIL import Image,ImageDraw,ImageOps,ImageFont

OUT=Path(__file__).resolve().parent
RUN=OUT.parent/'scene20/r20261019'
FONT=ImageFont.load_default(size=16)
def frame(path,slot):
    with av.open(str(path)) as c:
        c.streams.video[0].thread_count=1
        for i,f in enumerate(c.decode(video=0)):
            if i==slot:return f.to_image()
    raise ValueError((path,slot))

paths=sorted(RUN.glob('scenes/*/*/manifest.json'),key=lambda p:p.parent.name)
for page in range((len(paths)+9)//10):
    sheet=Image.new('RGB',(1320,5*350),'#20242b')
    for index,path in enumerate(paths[page*10:(page+1)*10]):
        p=path.parent;g=json.loads((p/'generation.json').read_text());count=g['output']['frames_per_video'];mid=(count-1)//2
        mapping=json.loads((p/'input/frame_mapping.json').read_text())['frames'][mid]
        ni=mapping['left_native_index'] if mapping['interpolation_fraction']<.5 else mapping['right_native_index']
        panels=[frame(p/'input/observed_native.mp4',ni),frame(p/'input/model_input.mp4',mid),frame(p/'videos/00.mp4',mid),frame(p/'videos/12.mp4',mid)]
        x=(index%2)*660;y=(index//2)*350;draw=ImageDraw.Draw(sheet)
        draw.text((x+4,y+4),f'{p.name} scene {p.parent.name} t={mid/25:.2f}s',fill='white',font=FONT)
        for j,(label,im) in enumerate(zip(['Native','Input','Generated 00','Generated 12'],panels)):
            draw.text((x+j*165+4,y+25),label,font=FONT,fill='white')
            im=ImageOps.contain(im,(165,300));sheet.paste(im,(x+j*165+(165-im.width)//2,y+48+(300-im.height)//2))
    sheet.save(OUT/f'appearance_survey_{page+1}.jpg',quality=94)
print(f'Surveyed {len(paths)} people at middle frame; views 00/12 only, no quality acceptance.')
