"""Segment person and vehicle independently with SAM2, then preserve their union."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, '/mnt/ojc/third_party/Grounded-SAM-2')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

items = json.loads(Path(sys.argv[1]).read_text())
predictor = SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',
    '/mnt/ojc/third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt', device='cuda'))
for item in items:
    out=Path(item['output']);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():continue
    try:
        rgb=np.array(Image.open(item['image']).convert('RGB'));h,w=rgb.shape[:2]
        masks=[];scores=[];boxes=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            predictor.set_image(rgb)
            for name in ('rider_box','vehicle_box'):
                box=np.array(item[name],dtype=float)*[w/1000,h/1000,w/1000,h/1000]
                if not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):raise ValueError('Invalid box')
                masks_raw,score,_=predictor.predict(box=box,multimask_output=False)
                mask=masks_raw[0].astype(bool)
                if mask.sum()<128:raise ValueError('Missing component mask')
                Image.fromarray((mask*255).astype('uint8')).save(out/(name+'_mask.png'))
                masks.append(mask);scores.append(float(score[0]));boxes.append(box)
        union=masks[0]|masks[1];ys,xs=np.where(union)
        x0=min(xs.min(),min(b[0] for b in boxes));y0=min(ys.min(),min(b[1] for b in boxes))
        x1=max(xs.max()+1,max(b[2] for b in boxes));y1=max(ys.max()+1,max(b[3] for b in boxes))
        pad=max(12,round(.06*max(x1-x0,y1-y0)))
        crop=[max(0,int(x0)-pad),max(0,int(y0)-pad),min(w,int(x1)+pad),min(h,int(y1)+pad)]
        rgba=Image.fromarray(np.dstack([rgb,(union*255).astype('uint8')])).crop(crop)
        rgba.save(out/'rgba.png')
        white=Image.new('RGB',rgba.size,'white');white.paste(rgba,mask=rgba.getchannel('A'));white.save(out/'anchor.png')
        Image.fromarray((union*255).astype('uint8')).save(out/'union_mask.png')
        result=dict(status='segmented',crop=crop,sam_scores=scores,component_pixels=[int(m.sum()) for m in masks],
                    union_pixels=int(union.sum()),rgba=str(out/'rgba.png'),anchor=str(out/'anchor.png'))
    except Exception as exc:result=dict(status='failed',error=repr(exc))
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(item['output'],result['status'],flush=True)
