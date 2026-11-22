"""Check whether temporal-change and lower-body proxies receive lower RGB weights."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from data import ROOT,save_json
p=argparse.ArgumentParser();p.add_argument('--object',choices=['R07','M02'],required=True);a=p.parse_args();oid={'R07':'OBJ_0018','M02':'OBJ_0019'}[a.object];clip=ROOT/'objects'/oid/'clips/C001';m=json.loads((clip/'manifest.json').read_text());protocol=m['protocols']['S1'];rows=[]
for view in protocol['train_camera_ids']:
 weights=np.load(clip/'05_reconstruction/confidence_S1'/f'cam{view:02d}.npz')['weights'].astype('float32');images=[];masks=[]
 for frame in protocol['train_frame_ids']:
  im=np.asarray(Image.open(Path(m['data_root'])/'images'/f'cam{view:02d}'/f'{frame:06d}.png'),dtype=np.float32)/255;images.append(im[...,:3]*im[...,3:]+.5*(1-im[...,3:]));masks.append(im[...,3]>.5)
 for i in range(1,len(images)-1):
  change=(np.abs(images[i]-images[i-1])+np.abs(images[i+1]-images[i])).mean(2)*.5;mask=masks[i];w=weights[i];q=np.quantile(change[mask],[.25,.75]);high=mask&(change>=q[1]);low=mask&(change<=q[0]);ys=np.where(mask)[0];lower=np.zeros_like(mask);lower[int(ys.min()+.65*(ys.max()-ys.min())):ys.max()+1]=True;lower&=mask
  rows.append({'view':view,'frame':protocol['train_frame_ids'][i],'foreground_weight':float(w[mask].mean()),'lower35pct_weight':float(w[lower].mean()),'high_temporal_change_weight':float(w[high].mean()),'low_temporal_change_weight':float(w[low].mean())})
keys=[k for k in rows[0] if k not in ['view','frame']];summary={k:float(np.mean([r[k] for r in rows])) for k in keys};save_json(clip/'08_evaluation/confidence_regions.json',{'summary':summary,'rows':rows,'scope':'Temporal RGB change and lower35% target-bbox proxies, not optical flow or annotated hands/feet. Lower weight can reflect actual motion, occlusion or inconsistent generated content; no claim which is correct. Full alpha supervision remains active.'});print(a.object,summary,flush=True)
