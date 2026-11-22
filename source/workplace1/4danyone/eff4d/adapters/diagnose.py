"""Diagnose native conditional covariance and unfiltered splat elongation; no model edits."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args()
import torch,numpy as np
from runtime import *
from gaussian_renderer import render as native
record=device_record();m,d=load(a.checkpoint);manifest=json.loads(Path(d['config']['manifest']).read_text());c=manifest['cameras'][8]
with torch.no_grad():
 cov,offset=m.get_current_covariance_and_mean_offset(timestamp=.5);mat=torch.zeros((len(cov),3,3),device='cuda');mat[:,0,0]=cov[:,0];mat[:,0,1]=mat[:,1,0]=cov[:,1];mat[:,0,2]=mat[:,2,0]=cov[:,2];mat[:,1,1]=cov[:,3];mat[:,1,2]=mat[:,2,1]=cov[:,4];mat[:,2,2]=cov[:,5]
 eig=torch.linalg.eigvalsh(mat);positive=eig[:,0]>0;ratio=torch.sqrt(eig[:,-1].clamp_min(0)/eig[:,0].clamp_min(1e-12));active=m.get_marginal_t(.5)[:,0]>.05
 def quant(x):return {str(q):float(torch.quantile(x.float(),q)) for q in [0.,.5,.9,.99,1.]}
 rows=[]
 for t in [0,.173,.5,1]:
  cuda=render(m,c,t);pipe=SimpleNamespace(**vars(PIPE));pipe.compute_cov3D_python=True
  python=native(Camera(c,t),m,pipe,torch.full((3,),.5,device='cuda'))
  rows.append({'time':t,'rgb_max_abs':float((cuda['render']-python['render']).abs().max()),'rgb_mean_abs':float((cuda['render']-python['render']).abs().mean()),'alpha_max_abs':float((cuda['alpha']-python['alpha']).abs().max())})
 result={'device':record,'checkpoint_sha256':sha(a.checkpoint),'conditional_covariance_positive_fraction':float(positive.float().mean()),'conditional_spatial_std_max_quantiles':quant(torch.sqrt(eig[:,-1].clamp_min(0))),'conditional_axis_ratio_quantiles':quant(ratio),'active_at_midtime':int(active.sum()),'active_axis_ratio_quantiles':quant(ratio[active]),'time_center_quantiles':quant(m._t.flatten()),'temporal_std_quantiles':quant(torch.sqrt(m.get_cov_t().flatten())),'cuda_vs_author_python_covariance':rows,'interpretation':'Large axis ratios may explain line-like splats; diagnostic association, not a causal ablation. No pruning/filtering performed.'}
 save_json(a.checkpoint.parent/'covariance_diagnostic.json',result);print(json.dumps(result),flush=True)
