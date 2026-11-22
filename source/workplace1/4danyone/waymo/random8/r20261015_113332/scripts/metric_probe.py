from pathlib import Path
import sys,os,json,hashlib
import numpy as np
from PIL import Image
import torch
from evaluate_branch import ssim_function,crop
r=Path(__file__).resolve().parents[1];data=r.parents[1]/'round2/reconstruction/data/m02_short61_24/images/cam00';a=np.asarray(Image.open(data/'000000.png'),np.float32)/255;b=np.asarray(Image.open(data/'000005.png'),np.float32)/255;x0,y0,x1,y1=crop([a[...,3]>.5]);ims=[v[...,:3]*v[...,3:]+.5*(1-v[...,3:]) for v in [a,b]];t=[torch.from_numpy(v.transpose(2,0,1).copy())[None] for v in ims];ssim,sh=ssim_function();scores={'ssim':float(ssim(*t)),'ssim_source_sha256':sh,'lpips_scope':'existing fixed real generated RGBA pair; CPU numerical compatibility only'}
if sys.argv[1]=='ftgs':
 from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
 model=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).eval()
 with torch.inference_mode():scores['lpips']=float(model(*[v[:,:,y0:y1,x0:x1] for v in t]))
else:
 import lpips
 model=lpips.LPIPS(net='alex').eval()
 with torch.inference_mode():scores['lpips']=float(model(*[v[:,:,y0:y1,x0:x1]*2-1 for v in t]))
(r/'research'/f'metric_probe_{sys.argv[1]}.json').write_text(json.dumps(scores,indent=2)+'\n');print(scores)
