"""Render fixed source camera and 16 shared source-relative azimuths on white."""
from pathlib import Path
import argparse,json,math,sys,hashlib,os
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
os.environ['PATH']='/home/wmj/miniconda3/envs/sim/bin:'+os.environ['PATH']
os.environ.setdefault('CPATH','/home/wmj/miniconda3/envs/splatad/include')
os.environ.setdefault('TORCH_EXTENSIONS_DIR',str(ROOT/'cache/torch_extensions'))
os.environ.setdefault('MAX_JOBS','4')
sys.path.insert(0,'/mnt/wmj/workspace4/pipeline')
from curate.core import load_rigid

def main():
 p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--variant',required=True);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
 source=ROOT/'inputs'/a.case/'evaluation/camera.json';cfg=json.loads(source.read_text());run=ROOT/'results'/a.case/f'{a.variant}_seed{a.seed}';asset=run/'asset'
 heading=json.loads((asset/'heading_review.json').read_text())
 if heading['front_after']!='+X':raise ValueError('Heading not verified')
 out=run/'evaluation';out.mkdir(exist_ok=True)
 gs=load_rigid(asset/'canonical.ply');xyz=gs.xyz[0].copy();extent=np.quantile(xyz[:,0],.995)-np.quantile(xyz[:,0],.005)
 factor=cfg['length_m']/extent
 arrays=[xyz*factor,gs.rot[0],gs.scale[0]*factor,gs.opacity,gs.color]
 if not all(np.isfinite(x).all() for x in arrays) or len(xyz)==0:raise ValueError('Invalid Gaussian payload')
 means,quats,scales,opacity,color=[torch.as_tensor(x,device='cuda',dtype=torch.float32) for x in arrays];quats/=quats.norm(dim=-1,keepdim=True)
 from gsplat import rasterization
 def render(v,k,w,h,name):
  with torch.no_grad():
   rgb,alpha,_=rasterization(means,quats,scales,opacity,color,torch.tensor(v[None],device='cuda',dtype=torch.float32),torch.tensor(k[None],device='cuda',dtype=torch.float32),w,h,backgrounds=torch.ones((1,3),device='cuda'),packed=False)
  rgb=rgb[0,:,:,:3].clamp(0,1).cpu().numpy();alpha=alpha[0,:,:,0].clamp(0,1).cpu().numpy()
  if not np.isfinite(rgb).all() or not np.isfinite(alpha).all():raise ValueError('Nonfinite render')
  count=int((alpha>.5).sum())
  Image.fromarray((rgb*255).round().astype('uint8')).save(out/(name+'.png'));Image.fromarray((alpha*255).round().astype('uint8')).save(out/(name+'_alpha.png'))
  return count
 counts=[];center=np.array([0.,0.,cfg['height_m']/2]);size=cfg['orbit_size'];f=cfg['orbit_focal_pixels'];k=np.array([[f,0,size/2],[0,f,size/2],[0,0,1.]])
 for j in range(16):
  az=cfg['orbit_phase_rad']+j*math.pi/8;el=math.radians(cfg['orbit_elevation_degrees']);eye=center+cfg['orbit_distance_m']*np.array([math.cos(az)*math.cos(el),math.sin(az)*math.cos(el),math.sin(el)])
  forward=(center-eye)/np.linalg.norm(center-eye);right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);down=np.cross(forward,right)
  r=np.stack([right,down,forward]);v=np.eye(4);v[:3,:3]=r;v[:3,3]=-r@eye
  counts.append(render(v,k,size,size,f'orbit_{j:02}'))
 matched=render(np.array(cfg['view_cv_from_object']),np.array(cfg['K_crop_undistorted']),*cfg['image_size'],'matched_source')
 record=dict(case_id=a.case,variant=a.variant,seed=a.seed,valid_export=True,valid_16_views=all(n>0 for n in counts),orbit_foreground_pixels=counts,matched_foreground_pixels=matched,metric_scale=factor,source_camera_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),camera_source=str(source),background='white',alpha_threshold=.5,posthoc_best_view_selection=False)
 (out/'render_audit.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
if __name__=='__main__':main()
