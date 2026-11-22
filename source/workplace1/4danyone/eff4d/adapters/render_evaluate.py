"""Independent checkpoint rendering and generated-observation evaluation."""
import argparse,json,time,subprocess,sys,contextlib,math
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--demo-only',action='store_true');args=p.parse_args()
import numpy as np,torch
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageOps
from runtime import *
from data import output_path,model_time,observation_seconds

class Video:
 def __init__(self,path,width,height,fps=25):
  self.path=output_path(path);self.width=width;self.height=height;self.frames=0;self.fps=fps
  self.proc=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-nostdin','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(self.path)],stdin=subprocess.PIPE)
 def add(self,arr):
  assert arr.shape==(self.height,self.width,3);self.proc.stdin.write(np.ascontiguousarray(arr,dtype='uint8').tobytes());self.frames+=1
 def close(self):
  self.proc.stdin.close();code=self.proc.wait();assert code==0,(self.path,code)
  decoded=imageio_ffmpeg.read_frames(str(self.path),pix_fmt='rgb24',input_params=['-threads','1']);meta=next(decoded);count=sum(1 for _ in decoded)
  assert count==self.frames and abs(meta['fps']-self.fps)<1e-3 and meta['size']==(self.width,self.height)
  return {'path':str(self.path),'frames':self.frames,'fps':self.fps,'sha256':sha(self.path),'full_decode':True}

def uint8(x):return np.round(np.clip(x,0,1)*255).astype('uint8')
def panel(image,text):
 im=Image.fromarray(image) if isinstance(image,np.ndarray) else image;canvas=Image.new('RGB',(352,680),(28,28,28));im=ImageOps.contain(im,(352,640));canvas.paste(im,((352-im.width)//2,40+(640-im.height)//2));ImageDraw.Draw(canvas).text((5,4),text,fill='white');return np.array(canvas)
def camera_orbit(cameras,num=120):
 from scipy.spatial.transform import Rotation,Slerp
 cs=sorted(cameras,key=lambda c:c['yaw']);angles=np.array([c['yaw'] for c in cs]+[360.]);poses=np.array([c['c2w'] for c in cs]+[cs[0]['c2w']]);slerp=Slerp(angles,Rotation.from_matrix(poses[:,:3,:3]));out=[]
 for angle in np.linspace(0,360,num,endpoint=False):
  c=dict(cs[0]);pose=np.eye(4);pose[:3,:3]=slerp(angle).as_matrix();pose[:3,3]=[np.interp(angle,angles,poses[:,k,3]) for k in range(3)];c['c2w']=pose.tolist();out.append(c)
 return out

def crop_bounds(masks):
 ys,xs=np.where(np.logical_or.reduce(masks));assert len(xs)>0;h,w=masks[0].shape;pad=max(4,round(.05*max(xs.max()-xs.min()+1,ys.max()-ys.min()+1)));return [max(0,int(xs.min())-pad),max(0,int(ys.min())-pad),min(w,int(xs.max())+pad+1),min(h,int(ys.max())+pad+1)]

start=time.monotonic();device=device_record();model,ck=load(args.checkpoint);manifest=Path(ck['config']['manifest']);m=json.loads(manifest.read_text());assert sha(manifest)==ck['config']['manifest_sha256'];out=output_path(args.checkpoint.parent/'evaluation'/'metrics.json').parent
if (out/'complete.json').exists():raise FileExistsError(out)
cams={c['id']:c for c in m['cameras']};views=m['validation_camera_ids'];eval_frames=m['evaluation_frame_ids'];eval_seconds=[observation_seconds(m,f) for f in eval_frames];
assert len(eval_frames)>1 and np.allclose(np.diff(eval_seconds),np.diff(eval_seconds)[0]),'Video comparison needs uniform evaluation time intervals'
eval_fps=1/float(np.diff(eval_seconds)[0]);train_frames=ck['config']['sampling']['train_frame_ids'];videos=[];rows=[]
metric=None
if not args.demo_only:
 import lpips
 from pytorch_msssim import ssim
 metric=lpips.LPIPS(net='alex',verbose=False).cuda().eval()
# Complete dynamic fixed-camera demo first. Every output frame is its real25fps timestamp.
writers={v:Video(out/f'view{v:02d}.mp4',704,680,fps=eval_fps) for v in views}
with torch.no_grad():
 for view in views:
  rgbs=[];masks=[]
  for frame in eval_frames:
   path=Path(m['data_root'])/'images'/f'cam{view:02d}'/f'{frame:06d}.png';assert sha(path)==m['source_hashes'][str(path.resolve())];a=np.asarray(Image.open(path),dtype=np.float32)/255;rgbs.append(a);masks.append(a[...,3]>.5)
  crop=crop_bounds([mask for f,mask in zip(eval_frames,masks) if f%5==0] or masks);x0,y0,x1,y1=crop
  for frame,rgba,mask in zip(eval_frames,rgbs,masks):
   sec=observation_seconds(m,frame)
   r=render(model,cams[view],model_time(m,sec),bg=0.);raw=r['render'];alpha=r['alpha'];rgb=torch.from_numpy(rgba[...,:3]).permute(2,0,1).cuda();ta=torch.from_numpy(rgba[...,3:]).permute(2,0,1).cuda();pred_white=(raw+1-alpha).clamp(0,1);target_white=rgb*ta+1-ta
   pred=pred_white.permute(1,2,0).cpu().numpy();target=target_white.permute(1,2,0).cpu().numpy();a=alpha[0].cpu().numpy()
   path=out/'renders'/f'cam{view:02d}'/f'{frame:06d}.png';path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(uint8(pred)).save(path);Image.fromarray(uint8(a)).save(path.with_name(f'{frame:06d}_alpha.png'))
   supervised=frame in train_frames;writers[view].add(np.concatenate([panel(uint8(target),f'Generated reference v{view:02d} / {sec:.2f}s'),panel(uint8(pred),f"{ck['config']['variant']} {ck['config']['steps']} steps / v{view:02d}\n{'Train-time query' if supervised else 'Unsupervised time query'}")],axis=1))
   if metric is not None:
    pg=(raw+.5*(1-alpha)).clamp(0,1);tg=rgb*ta+.5*(1-ta);err=(pg-tg).square().mean(0).cpu().numpy();sil=a>.5;union=(sil|mask).sum();outside=~mask;ys=np.where(mask)[0];lower=np.zeros_like(mask)
    if len(ys):lower[int(ys.min()+.65*(ys.max()-ys.min())):ys.max()+1]=True
    lower&=mask
    lp_white=float(metric(pred_white[None]*2-1,target_white[None]*2-1).item());lp_crop=float(metric(pg[None,:,y0:y1,x0:x1]*2-1,tg[None,:,y0:y1,x0:x1]*2-1).item())
    row={'view':view,'frame':frame,'seconds':sec,'supervised_time':supervised,'white_full_psnr':-10*math.log10(max(float((pred_white-target_white).square().mean()),1e-12)),'white_full_ssim':float(ssim(pred_white[None],target_white[None],data_range=1,size_average=True)),'white_full_alex_lpips':lp_white,'gray_foreground_psnr':-10*math.log10(max(float(err[mask].mean()),1e-12)),'gray_crop_alex_lpips':lp_crop,'silhouette_iou':float((sil&mask).sum()/max(union,1)),'outside_alpha_mean':float(a[outside].mean()),'outside_alpha_per_foreground_pixel':float(a[outside].sum()/max(mask.sum(),1)),'lower35pct_recall_alpha05':float((a[lower]>.5).mean()) if lower.any() else None,'crop':crop};rows.append(row)
   if frame in [0,30,60]:Image.fromarray(np.concatenate([uint8(target),uint8(pred),np.repeat(uint8(a)[...,None],3,2)],axis=1)).save(out/f'view{view:02d}_frame{frame:03d}.png')
  videos.append(writers[view].close());print('Rendered+evaluated view',view,flush=True)
 # Fixed front/side/back, actual action; camera never moves.
 writer=Video(out/'motion_fixed_views.mp4',1056,680,fps=m.get('render_fps',25))
 for sec in m['render_timestamps_sec']:
  panels=[]
  for view in [0,6,12]:
   r=render(model,cams[view],model_time(m,sec),bg=1.);panels.append(panel(uint8(r['render'].permute(1,2,0).cpu().numpy()),f'Fixed view{view:02d} / action {sec:.2f}s'))
  writer.add(np.concatenate(panels,1))
 videos.append(writer.close())
 # Fixed t=1.2seconds, moving camera; never claim this demonstrates motion.
 writer=Video(out/'geometry_orbit.mp4',352,680)
 for i,c in enumerate(camera_orbit(m['cameras'])):
  r=render(model,c,.5,bg=1.);writer.add(panel(uint8(r['render'].permute(1,2,0).cpu().numpy()),f"Geometry orbit / FIXED action t={m['model_time_to_real_seconds']['offset']+.5*m['model_time_to_real_seconds']['scale']:.2f}s\nCamera yaw {i*3} degrees"))
 videos.append(writer.close())
if rows:
 keys=[k for k in rows[0] if k not in ['view','frame','seconds','supervised_time','crop']]
 aggregates={}
 for name,subset in [('all61' if len(eval_frames)==61 else 'all_times',rows),('train_times',[r for r in rows if r['supervised_time']]),('interpolation_times',[r for r in rows if not r['supervised_time']]),('legacy13times',[r for r in rows if r['frame']%5==0])]:
  aggregates[name]={k:float(np.mean([r[k] for r in subset if r[k] is not None])) for k in keys}
 save_json(out/'metrics.json',{'state':'complete','device':device,'checkpoint_sha256':sha(args.checkpoint),'views':views,'frames':eval_frames,'rows':rows,'aggregates':aggregates,'protocol':{'resolution':[352,640],'backgrounds':'white full image; gray0.5 foreground/crop','LPIPS':'lpips0.1.4 pretrained AlexNet, input[-1,1], no resizing','SSIM':'pytorch_msssim1.0.0 Gaussian11 sigma1.5 data_range1','crop':'per-view target union of13 legacy evaluation times +5% margin; fixed across E0/E1, matching old gray crop rule','reference':'generated validation, not real unseen-view GT','confidence_exclusion':False,'filtering':False}})
save_json(out/'complete.json',{'state':'dynamic_rendering_and_evaluation_complete' if rows else 'dynamic_demo_complete_no_metrics','device':device,'videos':videos,'seconds':time.monotonic()-start,'checkpoint_sha256':sha(args.checkpoint),'raw_unfiltered':True,'visual_review':'pending','production_approved':False})
