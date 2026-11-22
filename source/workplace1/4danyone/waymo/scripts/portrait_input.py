"""Preserve pixels and time while preventing narrow center cropping of wide input."""
import json
import math
from pathlib import Path
import av
import numpy as np
from PIL import Image
from prepare import encode

BASE=Path(__file__).resolve().parents[1]


def main():
    src=BASE/'experiments/green_walk';dst=BASE/'experiments/green_portrait'
    dst.mkdir(exist_ok=False)
    for folder in ['inputs','research','logs','previews']:(dst/folder).mkdir()
    mapping=json.loads((src/'inputs/frame_mapping.json').read_text())
    with av.open(str(src/'inputs/main.mp4')) as con:
        stream=con.streams.video[0];stream.thread_count=1
        units=math.ceil(max(stream.width/11,stream.height/20))
        width,height=11*units,20*units
        if width%2:units+=1;width,height=11*units,20*units
        left=(width-stream.width)//2;top=(height-stream.height)//2
        padding=[left,top,width-stream.width-left,height-stream.height-top]
        images=[Image.fromarray(np.pad(f.to_ndarray(format='rgb24'),((padding[1],padding[3]),(padding[0],padding[2]),(0,0)),constant_values=255)) for f in con.decode(video=0)]
    encode(dst/'inputs/main.mp4',images,25)
    mapping['input_padding_ltrb']=padding
    mapping['background_compositing']['mask_source']='../green_walk/segmentation/target_masks.npz'
    mapping['adaptation']='Constant white padding to exact 11:20 input canvas, no source scaling, no time change, no motion recentering'
    T=np.eye(3);T[:2,2]=[left,top]
    for f in mapping['frames']:
        f['bbox_input_xyxy']=(np.array(f['bbox_input_xyxy'])+[left,top,left,top]).tolist()
        f['source_to_input']=(T@f['source_to_input']).tolist()
        f['K_input']=(T@f['K_input']).tolist()
    (dst/'inputs/frame_mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
    config=json.loads((src/'config.json').read_text());config.update(name='green_portrait',gpu=6,derived_from='green_walk',adaptation=mapping['adaptation'])
    (dst/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    images[0].thumbnail((352,640));images[0].save(dst/'previews/padded_input.jpg')
    print(json.dumps({'dimensions':[width,height],'padding':padding,'source':'green_walk','output':str(dst)}),flush=True)


if __name__=='__main__':main()
