"""Source-conditioned CLIP-I16, DINOv3-I16 and matched-source ED-R."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
CLIP_PATH=Path('/mnt/ojc/workplace1/4danyone/waymo/frontend_general_v1/text_motion/clip/ViT-B-32.pt')
DINO_PATH=ROOT/'cache/timm--vit_huge_plus_patch16_dinov3_qkvb.lvd1689m/model.safetensors'

def foreground(rgb_path,mask_path):
 rgb=np.asarray(Image.open(rgb_path).convert('RGB'));mask=np.asarray(Image.open(mask_path).convert('L'))>127
 if rgb.shape[:2]!=mask.shape:raise ValueError('RGB/mask shape mismatch')
 yy,xx=np.where(mask)
 if len(xx)==0:raise ValueError('Empty foreground')
 x0,x1=xx.min(),xx.max()+1;y0,y1=yy.min(),yy.max()+1
 side=int(np.ceil(max(x1-x0,y1-y0)*1.10));left=int(np.floor((x0+x1-side)/2));top=int(np.floor((y0+y1-side)/2))
 white=np.where(mask[:,:,None],rgb,255).astype('uint8');canvas=Image.new('RGB',(side,side),'white');canvas.paste(Image.fromarray(white),(-left,-top))
 alpha=Image.new('L',(side,side),0);alpha.paste(Image.fromarray(mask.astype('uint8')*255),(-left,-top))
 return canvas.resize((512,512),Image.Resampling.BICUBIC),np.asarray(alpha.resize((512,512),Image.Resampling.NEAREST))>127,dict(original_bbox=[int(x0),int(y0),int(x1),int(y1)],square_crop=[left,top,side],padding_per_side=.05,output_size=512)

class Encoders:
 def __init__(self,which):
  import torch
  self.torch=torch;self.which=which
  torch.backends.cuda.matmul.allow_tf32=False
  torch.backends.cudnn.allow_tf32=False
  if which=='clip':
   import clip
   self.model,self.preprocess=clip.load(str(CLIP_PATH),device='cuda',jit=False);self.model=self.model.float().eval()
  else:
   import timm
   from safetensors.torch import load_file
   self.model=timm.create_model('vit_huge_plus_patch16_dinov3_qkvb',pretrained=False,num_classes=0,img_size=512)
   self.model.load_state_dict(load_file(str(DINO_PATH)),strict=True)
   self.model.rope.periods=self.model.rope.periods.to(torch.bfloat16).to(torch.float32)
   self.model=self.model.cuda().eval()
 def encode(self,images,masks):
  import cv2
  t=self.torch
  if self.which=='clip':
   x=t.stack([self.preprocess(im) for im in images]).cuda()
   with t.inference_mode():v=self.model.encode_image(x).float()
   return dict(global_embedding=t.nn.functional.normalize(v,dim=-1).cpu().numpy())
  x=t.from_numpy(np.stack([np.array(im,dtype=np.float32)/255 for im in images])).permute(0,3,1,2).cuda()
  x=(x-t.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/t.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
  with t.inference_mode():tokens=self.model.forward_features(x)
  prefix=self.model.num_prefix_tokens;patch=tokens[:,prefix:];side=int(round(patch.shape[1]**.5))
  if side*side!=patch.shape[1]:raise ValueError('Non-square patch grid')
  pooled=[]
  for i,m in enumerate(masks):
   select=t.from_numpy(cv2.resize(m.astype('uint8'),(side,side),interpolation=cv2.INTER_NEAREST).reshape(-1)>0).cuda()
   if not select.any():raise ValueError('No foreground patch')
   pooled.append(patch[i,select].mean(0))
  return dict(global_embedding=t.nn.functional.normalize(tokens[:,0].float(),dim=-1).cpu().numpy(),foreground_embedding=t.nn.functional.normalize(t.stack(pooled).float(),dim=-1).cpu().numpy())

def main():
 p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--variants',nargs='+',default=['B0','R','F','H','B0_real1','B0_real4']);p.add_argument('--encoder',choices=['clip','dino'],required=True);p.add_argument('--seed',type=int,default=42);args=p.parse_args()
 model=Encoders(args.encoder);camera_path=ROOT/'inputs'/args.case/'evaluation/camera.json';camera=json.loads(camera_path.read_text())
 source,source_mask,source_crop=foreground(camera['source_rgb'],camera['source_mask']);source_feature=model.encode([source],[source_mask])
 for variant in args.variants:
  out=ROOT/'results'/args.case/f'{variant}_seed{args.seed}'/'evaluation';audit=json.loads((out/'render_audit.json').read_text())
  if not audit['valid_export'] or not audit['valid_16_views'] or audit['matched_foreground_pixels']==0:raise ValueError('Invalid render for '+variant)
  if audit['source_camera_sha256']!=hashlib.sha256(camera_path.read_bytes()).hexdigest():raise ValueError('Source camera changed')
  images=[];masks=[];crops=[]
  for name in [f'orbit_{i:02}' for i in range(16)]+['matched_source']:
   im,mask,crop=foreground(out/(name+'.png'),out/(name+'_alpha.png'));images.append(im);masks.append(mask);crops.append(crop)
  features=[]
  for start in range(0,17,2):features.append(model.encode(images[start:start+2],masks[start:start+2]))
  global_features=np.concatenate([f['global_embedding'] for f in features]);similarities=global_features[:16]@source_feature['global_embedding'][0]
  result=dict(case_id=args.case,variant=variant,seed=args.seed,encoder=args.encoder,source_id=camera['source_id'],source_crop=source_crop,render_crops=crops,view_cosines=similarities.tolist(),I16=float(similarities.mean()),ED_R=None,posthoc_best_view_selection=False,precision='float32',source_camera_sha256=audit['source_camera_sha256'])
  if args.encoder=='dino':
   fg=np.concatenate([f['foreground_embedding'] for f in features]);result['ED_R']=float(1-np.dot(fg[16],source_feature['foreground_embedding'][0]));result['pooling']='I16: final normalized CLS. ED-R: final foreground patch mean, no CLS/register tokens.'
   result['model_provenance']=json.loads((ROOT/'docs/dinov3_weights.json').read_text())
  else:result.update(model='OpenAI CLIP ViT-B/32',checkpoint=str(CLIP_PATH),encoder_input='Standard CLIP 224x224 transform applied to protocol 512x512 square foreground')
  result['torch_version']=model.torch.__version__
  (out/f'{args.encoder}_metrics.json').write_text(json.dumps(result,indent=2));print(args.case,variant,args.encoder,result['I16'],result['ED_R'],flush=True)
if __name__=='__main__':main()
