"""Complete fixed-time stage evidence; source and generated cameras remain explicitly unregistered."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageOps

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent/'random8/r20261015_113332'


def main():
    cv2.setNumThreads(2)
    stage=ROOT/'tmp/stage_contacts';stage.mkdir(exist_ok=False)
    records=[]
    for label in ['N01','N02','N03','N04','N06','N07','N08','N11']:
        sample=BASE/'samples'/label
        source=sample/'source';mapping=json.loads((source/'inputs/frame_mapping.json').read_text())['frames']
        with np.load(source/'segmentation/target_masks.npz') as z:masks=z['masks']
        n=len(mapping);assert len(masks)==n
        cap=cv2.VideoCapture(str(source/'visualization/source_chain.mp4'))
        sheet=Image.new('RGB',(1760,1992),'#333333');draw=ImageDraw.Draw(sheet)
        for row,frame in enumerate([0,(n-1)//2,n-1]):
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read();assert ok
            rgb=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));assert rgb.size==(1408,664)
            cells=[rgb.crop((i*352,24,(i+1)*352,664)) for i in range(4)]
            mask=Image.fromarray((masks[frame]>0).astype(np.uint8)*255).convert('RGB')
            canvas=Image.new('RGB',(352,640),'#808080');mask=ImageOps.contain(mask,(352,640));canvas.paste(mask,((352-mask.width)//2,(640-mask.height)//2))
            images=[cells[0],cells[1],canvas,cells[2],cells[3]]
            titles=['Observed crop; UNREGISTERED','Actual masked/RIFE input','Preprocessing binary mask','Condition; generated cam0','Generated reference cam0']
            for col,(image,title) in enumerate(zip(images,titles)):
                sheet.paste(image,(col*352,row*664+24));draw.text((col*352+4,row*664+5),f'{label} f{frame} | {title}',fill='white')
            records.append(dict(sample=label,frame=frame,time_seconds=frame/25,source_native_left_index=mapping[frame]['left_native_index'],
                source_registered_to_generated_camera=False,evidence_path=f'audit/stages/{label}.jpg',selection='fixed first/middle/last; no quality-based frame choice'))
        cap.release();sheet.save(stage/f'{label}.jpg',quality=95)
    (stage/'selection.json').write_text(json.dumps(records,indent=2)+'\n')
    final=ROOT/'audit/stages';assert not final.exists();stage.rename(final)
    print('All8 complete fixed-time stage contacts ready',flush=True)


if __name__=='__main__':main()
