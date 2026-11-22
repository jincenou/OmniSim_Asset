import sys,os,json,concurrent.futures
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
B=Path('/mnt/ojc/workplace1/human_cycle_update_20261108');O=B/'optimization_trial'
sys.path.insert(0,str(B));sys.argv=['probe','--root','/mnt/ojc/workplace1/asset_library/human','--hasim-root','/mnt/ojc/workplace1/Hasim_trajectory_assets','--output',str(O)]
import build_cycles as m
import numpy as np

def run(row):
 existing=O/f"{row['id']}.json"
 if existing.exists():
  r=json.loads(existing.read_text());return {'id':r['id'],'rescued':r['rescued']}
 a=m.AssetLibrary(m.ROOT).get(row['id']);rt=m.load_asset(a);ar=rt.arrays
 vis=np.flatnonzero(ar['opacities']>.2);ix=vis[np.linspace(0,len(vis)-1,min(512,len(vis)),dtype=int)]
 ts=ar['timestamps']-ar['timestamps'][0];tm=(ts[1:]+ts[:-1])/2;tm=tm[(tm>.15)&(tm<ts[-1]-.15)];xyz=m.positions(ar,tm,ix);w=ar['opacities'][ix].astype(float);w/=w.sum();z=np.median(xyz[:,:,2],axis=0);foot=z<np.quantile(z,.02)+.30;fw=w[foot]/w[foot].sum();pairs=[]
 for i,s in enumerate(tm):
  jj=np.flatnonzero((tm-s>=.64)&(tm-s<=min(2.4,ts[-1]-.3)))
  d=np.sum((xyz[jj]-xyz[i])**2,axis=-1);score=np.sqrt(d@w)/.06+np.sqrt(d[:,foot]@fw)/.09
  pairs.extend((float(c),float(s),float(tm[j])) for c,j in zip(score,jj))
 pairs.sort();rank=[]
 for _,s,e in pairs[:100]:
  for width in [.12,.16,.20,.24]:
   if s<width+.02 or e+width>rt.duration_s-.02 or e-s<=2*width:continue
   met=m.metrics(ar,s,e,width,ix);cost=max(met[k]/m.LIMITS[k] for k in met);rank.append((cost,s,e,width))
 rank.sort();tried=[];best=None
 for _,s,e,width in rank[:10]:
  met=m.metrics(ar,s,e,width,vis);p=m.CycleProfile(a.id,s,e,width,rt.duration_s,row['source_sha256'],'locomotion',a.native_speed_mps,qualified=True)
  c=m.CyclicRuntimeAsset(rt,p);f=[c.sample(p.period_s+d).means[vis].astype(float) for d in [-.001,0,.001]];ww=ar['opacities'][vis].astype(float);ww/=ww.sum();met['seam_velocity_error_mps']=float(np.sqrt(np.sum(((f[2]-2*f[1]+f[0])/.001)**2,axis=1)@ww));passed=all(met[k]<=lim for k,lim in m.LIMITS.items());tried.append({'profile':p.to_dict(),'metrics':met,'passed':passed})
  if passed:
   for t in [20.5,60.]:assert np.isfinite(c.sample(t).means).all()
   best=tried[-1];break
 result={'id':a.id,'old_metrics':row['metrics'],'rescued':best is not None,'best':best,'trials':tried,'limits_unchanged':m.LIMITS,'visual_review_complete':False}
 (O/f'{a.id}.json').write_text(json.dumps(result,indent=2));return {'id':a.id,'rescued':result['rescued']}
if __name__=='__main__':
 rows=[r for r in json.loads((B/'cycle_updates.json').read_text())['assets'] if r['status']=='needs_review'];rows.sort(key=lambda r:r['metrics']['body_rms_m']);selected=rows;(O/'selection.json').write_text(json.dumps([r['id'] for r in selected]))
 results=[]
 with concurrent.futures.ProcessPoolExecutor(4) as pool:
  for result in pool.map(run,selected):results.append(result);print(result,flush=True)
 (O/'summary.json').write_text(json.dumps({'selected':len(selected),'rescued':sum(r['rescued'] for r in results),'results':results,'source_assets_modified':False,'visual_review_complete':False},indent=2))
