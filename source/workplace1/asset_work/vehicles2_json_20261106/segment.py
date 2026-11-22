"""Segment exact projected annotation boxes, preserving camera/instance provenance."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
sys.path.insert(0,'/mnt/ojc/third_party/Grounded-SAM-2')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
items=json.loads(Path(sys.argv[1]).read_text())
predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml','/mnt/ojc/third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt',device='cuda'))
for a in items:
 out=Path(a['output']);out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():continue
 try:
  rgb=np.array(Image.open(a['frame']['path']).convert('RGB'));h,w=rgb.shape[:2]
  assert (w,h)==(1600,900),'Native image dimensions differ'
  box=np.array(a['bbox_xyxy'])
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   predictor.set_image(rgb);m,s,_=predictor.predict(box=box,multimask_output=False)
  mask=m[0].astype(bool)
  # Suppress spill outside the annotated target cuboid; neighbouring identities stay excluded.
  allowed=np.zeros_like(mask);x0,y0,x1,y1=np.round(box).astype(int);allowed[y0:y1+1,x0:x1+1]=True;mask &= allowed
  ys,xs=np.where(mask)
  if len(xs)<1000:raise ValueError('Insufficient foreground pixels')
  crop=[max(0,int(box[0])-12),max(0,int(box[1])-12),min(w,int(box[2])+13),min(h,int(box[3])+13)]
  rgba=Image.fromarray(np.dstack([rgb,(mask*255).astype('uint8')])).crop(crop);rgba.save(out/'rgba.png')
  white=rgb.copy();white[~mask]=255;Image.fromarray(white).crop(crop).save(out/'anchor.png')
  Image.fromarray(rgb).crop(crop).save(out/'context.jpg')
  result=dict(status='segmented',crop_box=crop,bbox_xyxy=box.tolist(),sam_score=float(s[0]),foreground_pixels=len(xs),rgba_path=str(out/'rgba.png'),anchor_path=str(out/'anchor.png'),context_path=str(out/'context.jpg'))
 except Exception as e:result=dict(status='failed',error=repr(e))
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(out,result['status'],flush=True)
