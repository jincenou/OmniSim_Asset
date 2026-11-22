"""RIFE temporal consistency on training observations only; RGB reliability weights."""
import argparse,json,sys,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--object',choices=['R07','M02']);p.add_argument('--manifest',type=Path);p.add_argument('--protocol',default='S1');args=p.parse_args()
import numpy as np,torch
import torch.nn.functional as F
from PIL import Image
from runtime import ROOT,device_record,loss_image
from data import sha,save_json,output_path,observation_seconds
sys.path.insert(0,str(ROOT/'repos/rife_local'))
from train_log.IFNet_HDv3 import IFNet
from author_ssim import ssim

def normalize(error):
 if not torch.isfinite(error).all():raise ValueError('Nonfinite disagreement')
 confidence=error.max()-error
 if float(confidence.max())<1e-8:return torch.ones_like(error),'constant_error_equal_reliability'
 return (confidence/confidence.max()).clamp(0,1),'inverted_max_normalized'

def tests():
 assert normalize(torch.zeros(4,5))[1]=='constant_error_equal_reliability'
 assert normalize(torch.full((4,5),.5))[0].min()==1
 try:normalize(torch.tensor([float('nan')]))
 except ValueError:pass
 else:raise AssertionError('Nonfinite not rejected')
 rgb=torch.ones(3,4,5)*.2;alpha=torch.zeros(1,4,5);x=torch.rand(3,4,5,requires_grad=True);pa=torch.rand(1,4,5,requires_grad=True)
 pred={'render':x,'alpha':pa};w=torch.linspace(.05,1,20).reshape(1,4,5).requires_grad_()
 l0,_=loss_image(pred,rgb,alpha);g0=torch.autograd.grad(l0,x,retain_graph=True)[0]
 l1,_=loss_image(pred,rgb,alpha,w);g1=torch.autograd.grad(l1,x,retain_graph=True)[0]
 assert not torch.allclose(g0,g1);assert torch.allclose(torch.autograd.grad(l0,pa,retain_graph=True)[0],torch.autograd.grad(l1,pa,retain_graph=True)[0]);l1.backward();assert w.grad is None
 return {'constant_error':True,'zero_denominator':True,'empty_mask_full_alpha':True,'nonfinite_rejected':True,'confidence_detached':True,'rgb_gradient_changed':float((g0-g1).abs().max()),'alpha_gradient_unchanged':True}

start=time.monotonic();device=device_record();manifest=args.manifest if args.manifest else ROOT/'objects'/({'R07':'OBJ_0018','M02':'OBJ_0019'}[args.object])/'clips/C001/manifest.json';m=json.loads(manifest.read_text());protocol=m['protocols'][args.protocol];out=output_path(manifest.parent/'05_reconstruction'/f'confidence_{args.protocol}'/'summary.json')
if out.exists():raise FileExistsError(out)
assert not set(protocol['train_camera_ids'])&set(m['validation_camera_ids'])
model=IFNet().eval().cuda();weights=ROOT/'repos/rife_local/train_log/flownet.pkl';state=torch.load(weights,map_location='cpu',weights_only=True);state={k.removeprefix('module.'):v for k,v in state.items()};removed=[k for k in state if k.startswith(('teacher.','caltime.'))];model.load_state_dict({k:v for k,v in state.items() if k not in removed},strict=True)
result={'state':'running','device':device,'tests':tests(),'manifest_sha256':sha(manifest),'weight_sha256':sha(weights),'rife_version':'4.25_0919, differs from author HDv3','removed_training_only_keys':removed,'protocol':protocol,'observations':'Selected generated RGBA composited on gray0.5; training times only','weight_rule':'mean(inverted RGB-L1, inverted author SSIM disagreement), detached; full-image normalization by sum weights','alpha':'unweighted full image','views':[]};save_json(out,result)
frames=protocol['train_frame_ids'];secs={f:observation_seconds(m,f) for f in m['available_frame_ids']}
with torch.no_grad():
 for view in protocol['train_camera_ids']:
  images=[];masks=[]
  for f in frames:
   path=Path(m['data_root'])/'images'/f'cam{view:02d}'/f'{f:06d}.png';assert sha(path)==m['source_hashes'][str(path.resolve())]
   rgba=np.array(Image.open(path));alpha=rgba[...,3:]/255;rgb=rgba[...,:3]/255*alpha+.5*(1-alpha)
   images.append(torch.from_numpy(rgb[...,::-1].copy()).permute(2,0,1).float()[None]);masks.append(rgba[...,3]>=128)
  maps=[];stats=[]
  for i,f in enumerate(frames):
   h,w=images[i].shape[-2:];current=images[i].cuda();neighbors=[];errors=[]
   if i in (0,len(frames)-1):weight=torch.ones((h,w),device='cuda');reason='endpoint_no_extrapolation';methods=['endpoint']
   else:
    for delta in [1,2]:
     if i-delta<0 or i+delta>=len(frames):continue
     left,right=frames[i-delta],frames[i+delta];ratio=(secs[f]-secs[left])/(secs[right]-secs[left]);assert 0<ratio<1
     a=F.pad(images[i-delta].cuda(),(0,(-w)%64,0,(-h)%64));b=F.pad(images[i+delta].cuda(),(0,(-w)%64,0,(-h)%64))
     pred=model(torch.cat([a,b],1),ratio,[16,8,4,2,1])[2][-1][...,:h,:w];assert torch.isfinite(pred).all()
     errors.append(((current-pred).abs().sum(1)[0],(1-ssim(current,pred,full=True,val_range=1).mean(1))[0]))
     neighbors.append({'left':left,'right':right,'ratio':ratio})
    rgb_error=torch.stack([e[0] for e in errors]).mean(0);ssim_error=torch.stack([e[1] for e in errors]).mean(0)
    wr,mr=normalize(rgb_error);ws,ms=normalize(ssim_error);weight=.5*(wr+ws);reason='RIFE_neighbor_prediction';methods=[mr,ms]
   assert weight.shape==(h,w) and torch.isfinite(weight).all() and weight.sum()>0
   v=weight.cpu().numpy();mask=masks[i];ys=np.where(mask)[0];lower=np.zeros_like(mask)
   if len(ys):lower[int(ys.min()+.65*(ys.max()-ys.min())):ys.max()+1]=True
   lower&=mask;stats.append({'frame':f,'neighbors':neighbors,'reason':reason,'normalizations':methods,'mean':float(v.mean()),'foreground_mean':float(v[mask].mean()) if mask.any() else None,'lower35pct_mean':float(v[lower].mean()) if lower.any() else None,'min':float(v.min()),'max':float(v.max())});maps.append(v.astype('float16'))
  path=out.parent/f'cam{view:02d}.npz';np.savez_compressed(path,weights=np.stack(maps),frames=np.array(frames));result['views'].append({'view':view,'path':str(path),'sha256':sha(path),'frames':stats});save_json(out,result);print(args.object,'view',view,'done',flush=True)
result.update(state='complete',seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated());save_json(out,result)
