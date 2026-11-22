"""Shared raw-render metric protocol and standalone dynamic replay for both families."""
import argparse,ast,hashlib,importlib.util,json,math,os,sys
from pathlib import Path
from types import SimpleNamespace
from contextlib import ExitStack
import numpy as np
from PIL import Image,ImageDraw
import torch
from scipy.spatial.transform import Rotation,Slerp
from all8_gpu_guard import validate_gpu
R=Path(__file__).resolve().parents[1];S=R/'scripts';BACK=R/'backend';F=int(os.environ['FDANYONE_FRAMES']);H=[0,8,16]
os.environ['TORCH_HOME']=str(R/'cache/torch');os.environ['MPLCONFIGDIR']=str(R/'cache/matplotlib')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
 t=p.with_suffix(p.suffix+'.writing');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def byte(v):return np.rint(np.clip(v,0,1)*255).astype(np.uint8)
def scalar(pred,target,alpha,mask):
 error=(pred-target)**2;s=alpha>.5;union=(s|mask).sum()
 return {'full_psnr':-10*math.log10(max(float(error.mean()),1e-12)),'foreground_psnr':-10*math.log10(max(float(error[mask].mean()),1e-12)),'silhouette_iou':float((s&mask).sum()/max(union,1)),'outside_alpha_mean':float(alpha[~mask].mean())}
def crop(masks):
 y,x=np.where(np.logical_or.reduce(masks));assert len(x)>100;pad=max(4,round(.05*max(x.max()-x.min()+1,y.max()-y.min()+1)));return [max(0,int(x.min())-pad),max(0,int(y.min())-pad),min(352,int(x.max())+pad+1),min(640,int(y.max())+pad+1)]
def ssim_function():
 # Execute the exact four official functions, avoiding an unused optional lpips import in the FTGS env.
 path=R/'source/4dgaussians/utils/loss_utils.py';tree=ast.parse(path.read_text());tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['gaussian','create_window','ssim','_ssim']];scope={'torch':torch,'F':torch.nn.functional,'Variable':torch.autograd.Variable,'exp':math.exp};exec(compile(tree,str(path),'exec'),scope);return scope['ssim'],sha(path)
def video_module():
 spec=importlib.util.spec_from_file_location('random8_video',BACK/'src/video_io.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.MP4Writer
class FT:
 def __init__(self,base,branch,manifest,snapshot=None):
  from evaluate_asset import Renderer,ply_readback
  self.base=base/branch;steps=snapshot if snapshot is not None else (5000 if branch=='ftgs_base5k' else 15000);self.checkpoint=self.base/'ckpts'/f'ckpt_{steps-1}.pt';self.renderer=Renderer(self.checkpoint,base/'data');self.ply=self.base/f'ckpt_{steps-1}.ftgs.ply';self.restored,errors,comments=ply_readback(self.ply,self.renderer.s);self.replay={'parameters':errors,'comments':comments,'render_errors':[]};self.config=self.renderer.cfg;self.manifest=manifest;self.parameters=0
 def render(self,view,frame,pose=None):
  from gsplat import rasterization
  from freetime_ops import positions_at_time,temporal_opacity
  if pose is None:return self.renderer.render(view,frame)
  s=self.renderer.s;cfg=self.renderer.cfg;t=frame/(F-1);camera=next(c for c in self.manifest['cameras'] if c['view']==view)
  rgb,a,_=rasterization(means=positions_at_time(s['means'],s['velocities'],s['times'],t,self.renderer.state['use_velocity']),quats=s['quats'],scales=s['scales'].exp(),opacities=(s['opacities'].sigmoid()*temporal_opacity(s['times'],s['durations'],t)).clamp_min(1e-4),colors=torch.cat([s['sh0'],s['shN']],1),viewmats=torch.linalg.inv(torch.as_tensor(pose[None],device='cuda',dtype=torch.float32)),Ks=torch.as_tensor(np.asarray(camera['K'])[None],device='cuda',dtype=torch.float32),width=352,height=640,sh_degree=cfg['sh_degree'],packed=cfg['packed'],near_plane=cfg['near_plane'],far_plane=cfg['far_plane'],rasterize_mode='antialiased' if cfg['antialiased'] else 'classic')
  return rgb[0].cpu().numpy(),a[0,...,0].cpu().numpy()
 def validate(self):
  for frame in [0,(F-1)//2,F-1]:
   a,b=self.renderer.render(0,frame);c,d=self.renderer.render(0,frame,self.restored);error=max(float(abs(a-c).max()),float(abs(b-d).max()));assert error<1e-5;self.replay['render_errors'].append(error)
 def asset(self):return {'format':'FreeTimeGsVanilla dynamic .ftgs.ply','dynamic_ply':str(self.ply),'ply_sha256':sha(self.ply),'checkpoint':str(self.checkpoint),'checkpoint_sha256':sha(self.checkpoint),'point_count':len(self.renderer.s['means']),'runtime':str(R/'source/ftgs'),'replay':self.replay}
class HUST:
 def __init__(self,base,branch,manifest,snapshot=None):
  sys.path.insert(0,str(R/'source/4dgaussians'));from scene.gaussian_model import GaussianModel
  from scene.waymo_generated import GeneratedCameras
  self.base=R/'four_dg/runs'/f'{base.name}_grid{branch[-2:]}';config=json.loads((self.base/'experiment.json').read_text());self.config=config;self.manifest=manifest;self.iteration=30000 if snapshot is None else snapshot;self.saved=self.base/f'point_cloud/iteration_{self.iteration}';self.model=GaussianModel(config['dataset']['sh_degree'],SimpleNamespace(**config['hidden']));self.model.load_ply(str(self.saved/'point_cloud.ply'));self.model.load_model(str(self.saved));self.model._deformation.eval();self.dataset=GeneratedCameras(manifest,H);self.pipe=SimpleNamespace(**config['pipeline']);self.replay={'render_errors':[]};self.parameters=sum(p.numel() for p in self.model._deformation.parameters())
 def render(self,view,frame,pose=None):
  from gaussian_renderer import render
  from scene.cameras import Camera
  camera=self.dataset[H.index(view)*F+frame]
  if pose is not None:
   w=np.linalg.inv(pose);template=camera;camera=Camera(colmap_id=view,R=w[:3,:3].T,T=w[:3,3],FoVx=template.FoVx,FoVy=template.FoVy,image=template.original_image,gt_alpha_mask=None,image_name='orbit',uid=template.uid,data_device='cpu',time=frame/(F-1));camera.projection_matrix=template.projection_matrix.clone();camera.full_proj_transform=camera.world_view_transform@camera.projection_matrix
  raw=render(camera,self.model,self.pipe,torch.zeros(3,device='cuda'),stage='fine',cam_type='WaymoGenerated')['render'];white=render(camera,self.model,self.pipe,torch.zeros(3,device='cuda'),stage='fine',cam_type='WaymoGenerated',override_color=torch.ones_like(self.model.get_xyz))['render'];assert float((white.max(0).values-white.min(0).values).max())<1e-5
  return raw.permute(1,2,0).cpu().numpy(),white[0].clamp(0,1).cpu().numpy()
 def validate(self):
  if self.iteration!=30000:
   self.replay['scope']='independent snapshot load; no saved training-time reference for intermediate pointcloud';return
  reference=np.load(self.base/'reference_render.npz')
  for frame in [0,(F-1)//2,F-1]:
   rgb,_=self.render(0,frame);error=float(abs(rgb.transpose(2,0,1)-reference[f'view0_frame{frame}']).max());assert error<1e-5,error;self.replay['render_errors'].append(error)
 def asset(self):return {'format':'HUST4DGaussians canonical PLY plus neural deformation','dynamic_files':{str(self.saved/name):sha(self.saved/name) for name in ['point_cloud.ply','deformation.pth','deformation_table.pth','deformation_accum.pth']},'config':str(self.base/'experiment.json'),'runtime':str(R/'source/4dgaussians'),'point_count':len(self.model.get_xyz),'deformation_parameters':self.parameters,'replay':self.replay}
@torch.inference_mode()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);parser.add_argument('--branch',required=True);parser.add_argument('--snapshot',type=int);arg=parser.parse_args();device=validate_gpu();base=R/'samples'/arg.label;manifest_path=R/'four_dg/data'/arg.label/'fdanyone.json';manifest=json.loads(manifest_path.read_text());assert F==manifest['frame_count'];out=base/'evaluations'/(arg.branch if arg.snapshot is None else f'{arg.branch}_snapshot{arg.snapshot}');out.mkdir(parents=True,exist_ok=False)
 isft=arg.branch.startswith('ftgs');model=FT(base,arg.branch,manifest,arg.snapshot) if isft else HUST(base,arg.branch,manifest,arg.snapshot);model.validate();ssim,ssim_sha=ssim_function();Writer=video_module()
 if arg.snapshot is not None:
  from compact_output import NullVideoWriter
  Writer=NullVideoWriter
 if isft:
  from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
  perceptual=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda().eval()
  def lp(a,b):
   value=float(perceptual(a,b));perceptual.reset();return value
  lpversion='torchmetrics AlexNet normalize=True; learned weights v0.1'
 else:
  import lpips
  perceptual=lpips.LPIPS(net='alex').cuda().eval()
  def lp(a,b):return float(perceptual(a*2-1,b*2-1))
  lpversion='lpips0.1.4 AlexNet v0.1; [-1,1]'
 sampled=np.rint(np.linspace(0,F-1,13)).astype(int).tolist();targets={}
 for view in H:
  masks=[np.asarray(Image.open(base/'data/images'/f'cam{view:02d}'/f'{f:06d}.png'))[...,3]>127 for f in sampled];targets[view]=crop(masks)
 rows=[];dense=[];snapshots=[];sha_inputs=[]
 with ExitStack() as stack:
  writers={v:stack.enter_context(Writer(str(out/f'view{v:02d}.mp4'),25,704,640)) for v in H};action=stack.enter_context(Writer(str(out/'fixed_action.mp4'),25,352,640));backgrounds=stack.enter_context(Writer(str(out/'backgrounds.mp4'),25,1056,640))
  for f in range(F):
   for view in H:
    path=base/'data/images'/f'cam{view:02d}'/f'{f:06d}.png';im=np.asarray(Image.open(path),np.float32)/255;mask=im[...,3]>.5;assert mask.sum()>100 and (~mask).any();raw,alpha=model.render(view,f);assert np.isfinite(raw).all() and np.isfinite(alpha).all();assert raw.shape==(640,352,3) and alpha.shape==(640,352)
    pred=np.clip(raw+(1-alpha[...,None])*.5,0,1);target=im[...,:3]*im[...,3:]+(1-im[...,3:])*.5;values=scalar(pred,target,alpha,mask);ys,_=np.where(mask);lower=mask&(np.arange(640)[:,None]>=ys.min()+.65*(ys.max()-ys.min()));values.update(view=view,frame=f,seconds=f/25,whole_target_recall_alpha05=float((alpha[mask]>.5).mean()),lower_proxy_recall_alpha05=float((alpha[lower]>.5).mean()),foreground_coverage=float(alpha[mask].mean()));dense.append(values.copy());writers[view].append_data(byte(np.concatenate([target,pred],1)))
    if view==0:
     action.append_data(byte(pred));bgimages=np.concatenate([np.clip(raw+(1-alpha[...,None])*b,0,1) for b in [0.,.5,1.]],1);backgrounds.append_data(byte(bgimages))
    if f in sampled:
     x0,y0,x1,y1=targets[view];a=torch.as_tensor(pred.transpose(2,0,1).copy(),device='cuda')[None];b=torch.as_tensor(target.transpose(2,0,1).copy(),device='cuda')[None];values.update(ssim=float(ssim(a,b)),crop_lpips=lp(a[:,:,y0:y1,x0:x1],b[:,:,y0:y1,x0:x1]),crop_xyxy=targets[view],input_sha256=sha(path),background=.5);rows.append(values);sha_inputs.append([view,f,values['input_sha256']])
    if arg.snapshot is None and f in [0,(F-1)//2,F-1]:
     pathout=out/f'view{view:02d}_frame{f:03d}.jpg';Image.fromarray(byte(np.concatenate([target,pred],1))).save(pathout,quality=95);snapshots.append(pathout.name)
     if view==0:Image.fromarray(byte(bgimages)).save(out/f'backgrounds_frame{f:03d}.jpg',quality=95)
  if arg.snapshot is None:
   # Fixed physical cameras interpolated on their actual ring; time remains midpoint.
   poses=np.asarray([c['camtoworld'] for c in manifest['cameras']]);closed=np.concatenate([poses,poses[:1]]);at=np.linspace(0,24,72,endpoint=False);rot=Slerp(np.arange(25),Rotation.from_matrix(closed[:,:3,:3]))(at).as_matrix();orbit=np.repeat(np.eye(4)[None],72,0);orbit[:,:3,:3]=rot
   orbit[:,:3,3]=np.stack([np.interp(at,np.arange(25),closed[:,axis,3]) for axis in range(3)],-1);writer=stack.enter_context(Writer(str(out/'orbit_fixed_time.mp4'),25,352,640))
   for pose in orbit:
    raw,alpha=model.render(0,(F-1)//2,pose);assert np.isfinite(raw).all() and np.isfinite(alpha).all();writer.append_data(byte(np.clip(raw+(1-alpha[...,None])*.5,0,1)))
 # Use dense diagnostic minima for details, never use heldout to select weights.
 worst=sorted(dense,key=lambda r:(r['silhouette_iou'],r['lower_proxy_recall_alpha05']))[:3];detailregions=[]
 for k,row in (enumerate(worst) if arg.snapshot is None else []):
  view,f=row['view'],row['frame'];im=np.asarray(Image.open(base/'data/images'/f'cam{view:02d}'/f'{f:06d}.png'),np.float32)/255;raw,alpha=model.render(view,f);pred=np.clip(raw+(1-alpha[...,None])*.5,0,1);target=im[...,:3]*im[...,3:]+(1-im[...,3:])*.5;x0,y0,x1,y1=targets[view];yb=int(y0+.65*(y1-y0));sheet=np.concatenate([target[yb:y1,x0:x1],pred[yb:y1,x0:x1]],1);Image.fromarray(byte(sheet)).save(out/f'anomaly{k}_lower_proxy.png');detailregions.append({'kind':'target union bbox bottom35% proxy; not semantic foot GT','view':view,'frame':f,'xyxy':[x0,yb,x1,y1]})
 import imageio.v2 as imageio
 videos={}
 for path in out.glob('*.mp4'):
  reader=imageio.get_reader(str(path),input_params=['-threads','1'])
  try:count=sum(1 for _ in reader)
  finally:reader.close()
  assert count==(72 if path.name=='orbit_fixed_time.mp4' else F);videos[path.name]={'frames':count,'sha256':sha(path)}
 keys=['full_psnr','ssim','foreground_psnr','crop_lpips','silhouette_iou','outside_alpha_mean'];asset=model.asset();save(out/'asset_manifest.json',{**asset,'source_manifest':str(manifest_path),'source_manifest_sha256':sha(manifest_path),'coordinate_mapping':manifest['source_to_normalized'],'frame_count':F,'fps':25,'source_to_world_verified':False,'arbitrary_smplx_retargeting':False,'production_approved':False,'new_process_pid':os.getpid(),'device':device})
 result={'state':'new_process_replayed_evaluated_decoded_visual_review_pending','label':arg.label,'branch':arg.branch,'snapshot':arg.snapshot,'nominal_total_iterations':(arg.snapshot if isft else 3000+arg.snapshot) if arg.snapshot is not None else (5000 if arg.branch=='ftgs_base5k' else 15000 if isft else 33000),'device':device,'mean':{k:float(np.mean([r[k] for r in rows])) for k in keys},'samples':rows,'dense_anomaly_samples':dense,'worst_diagnostic_frames':worst,'detail_regions':detailregions,'sampled_frames':sampled,'heldout_views':H,'crops':targets,'videos':videos,'snapshots':snapshots,'replay':model.replay,'production_approved':False,'quality':{'technical':'passed','generated':'pending','reconstruction_visual':'pending','scene_ready':False},'protocol':{'name':'random8-v1.1-generated-heldout','resolution':[352,640],'background':.5,'reference':'generated-heldout reconstruction; not real Waymo novel-view GT','lpips':lpversion,'ssim':'exact official HUST window11 sigma1.5, zero padding, C1=.01^2,C2=.03^2','ssim_source_sha256':ssim_sha,'foreground_psnr':'target saved alpha>0.5; no prediction clipping by target mask','crop':'union of target masks across same13 times perview +5% or4px;no resizing','composition':'straight targetRGB once byalpha; predicted rasterRGB already premultiplied','time':'f/(F-1); real duration=(F-1)/25 seconds','lower_region':'bbox bottom35% proxy; no semantic hand/foot/accessory truth','all_input_pairs_sha256':hashlib.sha256(json.dumps(sha_inputs).encode()).hexdigest()}}
 result['output_retention']='final_full_visualization' if arg.snapshot is None else 'budget_metrics_only_v2'
 if arg.snapshot is not None:result['state']='new_process_replayed_budget_metrics_visuals_omitted'
 save(out/'results.json',result);print(json.dumps({'state':result['state'],'branch':arg.branch,'mean':result['mean']}),flush=True)
async def budgets():
 import asyncio
 parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);parser.add_argument('--branch',required=True);args=parser.parse_args();snapshots=[] if args.branch=='ftgs_base5k' else [5000,10000] if args.branch=='ftgs_opt15k' else [12000];sem=asyncio.Semaphore(1)
 async def execute(snapshot):
  async with sem:
   path=R/'samples'/args.label/'evaluations'/f'{args.branch}_snapshot{snapshot}'
   if (path/'results.json').exists():return
   log=R/'logs'/f'{args.label}_{args.branch}_budget{snapshot}.log'
   with log.open('xb') as output:
    child=await asyncio.create_subprocess_exec(sys.executable,'-u',__file__,'--label',args.label,'--branch',args.branch,'--snapshot',str(snapshot),stdout=output,stderr=asyncio.subprocess.STDOUT);record={'pid':child.pid,'snapshot':snapshot,'state':'running','log':str(log)};receipt=log.with_suffix('.json');save(receipt,record);code=await child.wait();record.update(state='complete' if code==0 else 'failed',exit_code=code);save(receipt,record)
   if code:raise RuntimeError(f'Budget evaluation failed:{snapshot};{log}')
 await asyncio.gather(*map(execute,snapshots))
if __name__=='__main__':
 import asyncio
 main()
 if '--snapshot' not in sys.argv:
  import gc
  gc.collect();torch.cuda.empty_cache();asyncio.run(budgets())
