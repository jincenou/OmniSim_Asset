"""SAM2 segmentation of reviewed web vehicle boxes; no fabricated cameras."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
sys.path.insert(0,'/mnt/ojc/third_party/Grounded-SAM-2')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
packet=Path(sys.argv[1]);items=json.loads(packet.read_text())
predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml','/mnt/ojc/third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt',device='cuda'))
for item in items:
 out=Path(item['image_path']).parent/'segmentation';out.mkdir(exist_ok=True)
 if (out/'result.json').exists():continue
 rgb=np.array(Image.open(item['image_path']).convert('RGB'));h,w=rgb.shape[:2]
 box=np.array(item['review']['bbox_xyxy_1000'],dtype=float)*[w/1000,h/1000,w/1000,h/1000]
 try:
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   predictor.set_image(rgb);m,s,_=predictor.predict(box=box,multimask_output=False)
  mask=m[0].astype(bool);ys,xs=np.where(mask)
  if len(xs)<2048:raise ValueError('Insufficient foreground pixels')
  crop=[max(0,int(min(xs.min(),box[0]))-24),max(0,int(min(ys.min(),box[1]))-24),min(w,int(max(xs.max()+1,box[2]))+24),min(h,int(max(ys.max()+1,box[3]))+24)]
  Image.fromarray((mask*255).astype('uint8')).save(out/'mask.png')
  Image.fromarray(np.dstack([rgb,(mask*255).astype('uint8')])).crop(crop).save(out/'rgba.png')
  white=rgb.copy();white[~mask]=255;Image.fromarray(white).crop(crop).save(out/'anchor.png')
  overlay=rgb.copy();overlay[mask]=(overlay[mask]*.6+np.array([50,220,160])*.4).astype('uint8');Image.fromarray(overlay).save(out/'overlay.jpg')
  result=dict(status='segmented',crop_box=crop,bbox_xyxy=box.tolist(),sam_score=float(s[0]),foreground_pixels=len(xs),rgba_path=str(out/'rgba.png'),anchor_path=str(out/'anchor.png'))
 except Exception as e:result=dict(status='failed',error=repr(e))
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(item['source_id'],result['status'],flush=True)
