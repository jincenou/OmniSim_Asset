from pathlib import Path
import sys,os,json,argparse
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluate_pixel_metrics import compute,foreground
import torch,lpips
R=Path(__file__).resolve().parents[2]/'nonvehicle10';p=argparse.ArgumentParser();p.add_argument('--cases',nargs='+',required=True);a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
net=lpips.LPIPS(net='alex',version='0.1',lpips=True,pnet_rand=False,pnet_tune=False,verbose=False).cuda().eval()
for cid in a.cases:
 case=json.loads((R/'inputs'/cid/'segmented_manifest.json').read_text());src=next(r for r in case['images'] if r['id']==case['flux_reference_id']);im,mask,_=foreground(src['path'],src['mask_path'])
 for variant in ['B0','R','H','B0_real1']:
  out=R/'results'/cid/f'{variant}_seed42';render=out/'orbit'
  if not (render/'render_config.json').exists():continue
  rows=[]
  for i in range(16):
   ri,rm,_=foreground(render/f'view_{i:02}.png',render/f'alpha_{i:02}.png');rows.append(compute(im,mask,ri,rm,net))
  mean={k:float(np.mean([r[k] for r in rows])) for k in rows[0]};result=dict(case_id=cid,variant=variant,protocol='Orbit-mean source-to-render appearance diagnostics, independent foreground normalization, all16views, NOT matched-source reconstruction accuracy',view_scores=rows,mean=mean)
  (out/'pixel_metrics.json').write_text(json.dumps(result,indent=2));print(cid,variant,mean,flush=True)
