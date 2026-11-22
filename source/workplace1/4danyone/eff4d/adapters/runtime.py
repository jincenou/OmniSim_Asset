"""Native Fudan generation 4DGS runtime with calibrated CV cameras and explicit time."""
import json,math,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from data import ROOT,save_json,sha
REPO=ROOT/'repos/4d-gaussian-splatting'
sys.path.insert(0,str(REPO))
SPEC={'sh_degree':0,'gaussian_dim':4,'time_duration':[0.,1.],'rot_4d':True,'force_sh_3d':True,'sh_degree_t':0}
PIPE=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,env_map_res=0,debug=False)

def device_record():
 requested=os.environ.get('CUDA_VISIBLE_DEVICES','')
 assert requested in ['4','5','6','7'],f'Single physical GPU4–7 required, got {requested}'
 physical=int(requested)
 expected=subprocess.check_output(['nvidia-smi',f'--id={physical}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 import ctypes
 cudart=torch.cuda.cudart();assert torch.cuda.device_count()==1
 # CUDA driver UUID queried for actual logical device, independent of the label.
 lib=ctypes.CDLL('libcuda.so.1');assert lib.cuInit(0)==0
 dev=ctypes.c_int();assert lib.cuDeviceGet(ctypes.byref(dev),0)==0
 raw=(ctypes.c_ubyte*16)();assert lib.cuDeviceGetUuid(ctypes.byref(raw),dev)==0
 import uuid
 actual='GPU-'+str(uuid.UUID(bytes=bytes(raw)));assert actual==expected,(actual,expected)
 return {'pid':os.getpid(),'physical_gpu':physical,'cuda_uuid':actual,'torch':torch.__version__,'cuda':torch.version.cuda,'name':torch.cuda.get_device_name(0)}

class Camera:
 def __init__(self,c,time):
  K=np.asarray(c['K']);W,H=c['width'],c['height'];assert K[0,1]==0 and K[1,0]==0
  self.image_width=W;self.image_height=H;self.FoVx=2*math.atan(W/(2*K[0,0]));self.FoVy=2*math.atan(H/(2*K[1,1]));self.timestamp=float(time)
  c2w=np.asarray(c['c2w']);w2c=np.linalg.inv(c2w);near,far=.01,100.
  P=np.zeros((4,4));P[0,0]=2*K[0,0]/W;P[1,1]=2*K[1,1]/H
  # Rasterizer ndc2Pix(v,S)=((v+1)*S-1)/2; preserve exact pixel principal point.
  P[0,2]=(2*K[0,2]+1)/W-1;P[1,2]=(2*K[1,2]+1)/H-1
  P[2,2]=far/(far-near);P[2,3]=-near*far/(far-near);P[3,2]=1
  self.world_view_transform=torch.tensor(w2c.T,dtype=torch.float32,device='cuda')
  self.projection_matrix=torch.tensor(P.T,dtype=torch.float32,device='cuda')
  self.full_proj_transform=self.world_view_transform@self.projection_matrix
  self.camera_center=torch.tensor(c2w[:3,3],dtype=torch.float32,device='cuda')

def arguments(steps):return SimpleNamespace(percent_dense=.05,position_lr_init=.001,position_lr_final=.00002,position_lr_delay_mult=.02,position_lr_max_steps=steps,position_t_lr_init=.001,feature_lr=.01,opacity_lr=.05,scaling_lr=.005,rotation_lr=.005)

def create(init,steps,seed=20261014):
 from scene.gaussian_model import GaussianModel
 from utils.graphics_utils import BasicPointCloud
 torch.manual_seed(seed);np.random.seed(seed)
 d=np.load(init);m=GaussianModel(**SPEC)
 m.create_from_pcd(BasicPointCloud(points=d['points'],colors=d['colors'],normals=np.zeros_like(d['points']),time=d['time']),1.)
 # Multi-time hulls contain repeated xyz; spatial-only KNN otherwise collapses scales.
 with torch.no_grad():m._scaling.fill_(math.log(.022));m._scaling_t.fill_(math.log(.15))
 m.training_setup(arguments(steps));return m

def render(model,c,time,bg=.5):
 from gaussian_renderer import render as upstream_render
 return upstream_render(Camera(c,time),model,PIPE,torch.full((3,),bg,device='cuda',dtype=torch.float32))

def load(path,training=False):
 from scene.gaussian_model import GaussianModel
 d=torch.load(path,map_location='cuda',weights_only=False);m=GaussianModel(**d['model_spec']);m.restore(d['capture'],arguments(d['config']['steps']) if training else None);return m,d

def checkpoint(path,m,config,step,rng,extra=None):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 payload={'capture':m.capture(),'model_spec':SPEC,'config':config,'step':step,'rng_sampling':rng.bit_generator.state,'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'extra':extra or {}}
 torch.save(payload,p.with_suffix('.tmp'));os.replace(p.with_suffix('.tmp'),p)
 save_json(p.parent/'model_spec.json',SPEC)

def loss_image(pred,rgb,alpha,confidence=None,bg=.5):
 target=rgb*alpha+bg*(1-alpha)
 weight=torch.ones_like(alpha) if confidence is None else confidence.detach()
 if not torch.isfinite(weight).all() or float(weight.sum())<=0:raise ValueError('Invalid confidence weights')
 rgb_loss=((pred['render']-target).square()*weight).sum()/(3*weight.sum())
 alpha_loss=(pred['alpha']-alpha).square().mean()
 return rgb_loss+alpha_loss,{'rgb':float(rgb_loss.detach()),'alpha':float(alpha_loss.detach())}
