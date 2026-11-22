"""Controlled background-removal comparison using existing verified masks."""
import json
import os
from pathlib import Path
import av
import numpy as np
from PIL import Image
from prepare import encode

BASE=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get('FDANYONE_WAYMO_ROOT',str(BASE))).resolve()


def main():
    source=ROOT/'inputs_rife'
    masks=np.load(ROOT/'segmentation/target_masks.npz')['masks']
    frames=[]
    with av.open(str(source/'main.mp4')) as con:
        con.streams.video[0].thread_count=1
        for i,frame in enumerate(con.decode(video=0)):
            rgb=frame.to_ndarray(format='rgb24')
            alpha=masks[i].astype(np.float32)[...,None]/255
            composite=np.rint(rgb*alpha+255*(1-alpha)).clip(0,255).astype(np.uint8)
            frames.append(Image.fromarray(composite))
    assert len(frames)==21
    encode(ROOT/'inputs/main.mp4',frames,25)
    mapping=json.loads((source/'frame_mapping.json').read_text())
    mapping['background_compositing']={'method':'straight alpha over constant white','mask_source':'segmentation/target_masks.npz',
       'purpose':'isolate the selected pedestrian from other people and background','model':'existing SAM2.1 Hiera Large with explicit target prompts'}
    for item in mapping['frames']:item['background_composited']=True
    (ROOT/'inputs/frame_mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
    print('Prepared foreground-controlled comparison; identity, geometry and time unchanged.',flush=True)


if __name__=='__main__':main()
