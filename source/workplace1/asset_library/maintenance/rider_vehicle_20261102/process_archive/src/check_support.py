"""Geometric support proxy; does not claim semantic wheel or scene-contact verification."""
from pathlib import Path
import sys,json,numpy as np
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'src/DriveLab'))
from asset_extraction.gaussians import load_rigid
for n in sys.argv[1:]:
 g=load_rigid(W/'runs'/n/'selected/canonical.ply');v=g.xyz[0];lo,hi=np.quantile(v,[.005,.995],axis=0);prior=json.loads((W/'configs/metric_priors.json').read_text())[n];s=prior['length_m']/(hi[0]-lo[0]);res={}
 for label,mask in [('front',v[:,0]>.2*(hi[0]-lo[0])),('rear',v[:,0]<-.2*(hi[0]-lo[0]))]:
  z=v[mask&(g.opacity>.5)&(np.abs(v[:,1])<.15*(hi[0]-lo[0])),2];res[label]=float(np.quantile(z,.005)*s)
 res.update(dimensions_lwh_m=((hi-lo)*s).tolist(),support_difference_m=abs(res['front']-res['rear']),method='Central lateral strips at both longitudinal ends; opacity > 0.5 and lower 0.5% z quantile. Geometric proxy, not semantic wheel segmentation.',metric_scale=float(s))
 (W/'runs'/n/'selected/support_validation.json').write_text(json.dumps(res,indent=2));print(n,res)
