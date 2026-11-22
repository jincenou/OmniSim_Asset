"""Make readable target-centered inspection sheets from the actual experiment."""
import json
import os
from pathlib import Path

import av
import numpy as np
from PIL import Image,ImageDraw,ImageOps

ROOT=Path(os.environ['FDANYONE_WAYMO_ROOT']).resolve()


def main():
    mapping=json.loads((ROOT/'inputs/frame_mapping.json').read_text())
    masks=np.load(ROOT/'segmentation/target_masks.npz')['masks']
    picked={0,7,15,22,30,37,45,52,60};canvas=Image.new('RGB',(9*176,2*344),'#222222');draw=ImageDraw.Draw(canvas)
    with av.open(str(ROOT/'inputs_rife/main.mp4')) as con:
        con.streams.video[0].thread_count=1
        for i,frame in enumerate(con.decode(video=0)):
            if i not in picked:continue
            rgb=frame.to_ndarray(format='rgb24');alpha=masks[i][...,None].astype(np.float32)/255
            composite=np.rint(rgb*alpha+255*(1-alpha)).clip(0,255).astype(np.uint8)
            b=np.array(mapping['frames'][i]['bbox_input_xyxy'])+[-16,-16,16,16]
            b=tuple(map(int,b));col=sorted(picked).index(i)
            for row,im in enumerate((Image.fromarray(rgb),Image.fromarray(composite))):
                im=ImageOps.contain(im.crop(b),(172,320));canvas.paste(im,(176*col+(176-im.width)//2,344*row+24))
                draw.text((176*col+4,344*row+4),f'{"source" if row==0 else "SAM2"} f{i}',fill='white')
    canvas.save(ROOT/'previews/foreground_review.jpg',quality=94)
    print(ROOT/'previews/foreground_review.jpg',flush=True)


if __name__=='__main__':main()
