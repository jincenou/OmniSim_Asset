"""SAM2 rider/vehicle box prompts, separate masks and lossless union evidence."""
import argparse,sys,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
sys.path.insert(0,'/mnt/ojc/third_party/Grounded-SAM-2')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
p=argparse.ArgumentParser();p.add_argument('packet',type=Path);args=p.parse_args()
x=json.loads(args.packet.read_text()); root=args.packet.parent
model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml','/mnt/ojc/third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt',device='cuda')
predictor=SAM2ImagePredictor(model)
for row in x['observations']:
 out=root/row['id']
 if out.exists():raise FileExistsError(f'Preserve prior segmentation evidence: {out}; use a new packet directory')
 out.mkdir();rgb=np.array(Image.open(row['image_path']).convert('RGB'))
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  predictor.set_image(rgb)
  masks=[];all_scores=[]
  for name in ['rider','vehicle']:
   print('INFERENCE_START SAM2',row['id'],name,flush=True)
   points=row.get(name+'_points'); labels=row.get(name+'_labels')
   m,scores,_=predictor.predict(box=np.array(row[name+'_box']),point_coords=np.array(points) if points else None,point_labels=np.array(labels) if labels else None,multimask_output=False)
   mask=m[0].astype(bool)
   for part in row.get(name+'_parts',[]):
    pm,_,_=predictor.predict(box=np.array(part),multimask_output=False)
    mask |= pm[0].astype(bool)
   all_scores.append(float(scores[0]));masks.append(mask);Image.fromarray((mask*255).astype('uint8')).save(out/f'{name}_mask.png')
  union=masks[0]|masks[1];Image.fromarray((union*255).astype('uint8')).save(out/'union_mask.png')
  rgba=np.dstack([rgb,(union*255).astype('uint8')]);ys,xs=np.where(union);box=[max(0,int(xs.min())-20),max(0,int(ys.min())-20),min(rgb.shape[1],int(xs.max())+21),min(rgb.shape[0],int(ys.max())+21)]
  Image.fromarray(rgba).crop(box).save(out/'rgba.png');white=rgb.copy();white[~union]=255;Image.fromarray(white).crop(box).save(out/'anchor.png')
  overlay=rgb.copy();overlay[union]=(overlay[union]*.55+np.array([40,220,180])*.45).astype('uint8');Image.fromarray(overlay).crop(box).save(out/'overlay.jpg')
  row.update(rgba_path=str(out/'rgba.png'),anchor_path=str(out/'anchor.png'),crop_box=box,sha256=hashlib.sha256((out/'rgba.png').read_bytes()).hexdigest(),sam2_scores=all_scores,mask_pixels=[int(m.sum()) for m in masks],union_pixels=int(union.sum()))
(root/'segmentation.json').write_text(json.dumps(x,indent=2))
print('SAM2_DONE',flush=True)
