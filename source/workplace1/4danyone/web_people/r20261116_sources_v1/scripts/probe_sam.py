from pathlib import Path
import sys,json
import numpy as np
import torch
from PIL import Image
from prepare_person import ROOT,PROJECT
sys.path.insert(0,str(PROJECT/'waymo/segmentation'))
from sam2.build_sam import build_sam2_video_predictor
from sam_transfer_guard import install_transfer_guard

torch.set_num_threads(2)
dest=ROOT/'inputs/QWH0003'
mapping=json.loads((dest/'inputs/frame_mapping.json').read_text())
box=np.asarray(mapping['frames'][0]['bbox_input_xyxy'],np.float32)
x0,y0,x1,y1=box;bw,bh=x1-x0,y1-y0
points=np.asarray([[x0+.5*bw,y0+.15*bh],[x0+.5*bw,y0+.45*bh],[x0+.45*bw,y0+.8*bh]],np.float32)
out=ROOT/'reviews/sam_probe';out.mkdir(exist_ok=True)
for offload in [False,True]:
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  p=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml',str(PROJECT/'waymo/segmentation/sam2.1_hiera_large.pt'),device='cuda',apply_postprocessing=False)
  install_transfer_guard(p)
  state=p.init_state(video_path=str(dest/'segmentation/frames'),offload_video_to_cpu=True,offload_state_to_cpu=offload)
  _,_,logits=p.add_new_points_or_box(state,frame_idx=0,obj_id=1,points=points,labels=np.ones(3,np.int32),box=box)
  alpha=logits[0,0].sigmoid().float().cpu().numpy()
  Image.fromarray((alpha*255).astype('uint8')).save(out/f'alpha_offload_{offload}.png')
  rgb=np.array(Image.open(dest/'segmentation/frames/00000.jpg'))
  Image.fromarray(np.rint(rgb*alpha[...,None]+255*(1-alpha[...,None])).clip(0,255).astype('uint8')).save(out/f'composite_offload_{offload}.jpg')
 del state,p;torch.cuda.empty_cache()
