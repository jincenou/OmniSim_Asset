from pathlib import Path
import csv,json,math,sys
import numpy as np
D=Path(__file__).resolve().parents[1];OLD=D.parent/'early_visual_query_20261019'
def load(p):
 rows=list(csv.DictReader(p.open()));out=[]
 for r in rows:
  for k,v in list(r.items()):
   if v=='':r[k]=None
   elif v in ['True','False']:r[k]=v=='True'
   else:
    try:r[k]=float(v)
    except ValueError:pass
  out.append(r)
 return out

def write(p,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=keys);w.writeheader();w.writerows(rows)

def avg(rows,k,w=None):
 x=[r for r in rows if r.get(k) is not None]
 if not x:return None
 return float(np.average([r[k] for r in x],weights=[r[w] for r in x] if w else None)) if not w or sum(r[w] for r in x)>0 else None

def aggregate(rows):
 good=[r for r in rows if r.get('known',True)];out=dict(total_rows=len(rows),known_rows=len(good),unknown_rows=len(rows)-len(good),pixels=sum(r.get('pixels',0) for r in good),reference_alpha_mass=sum(r.get('reference_alpha_mass',0) for r in good),temporal_valid_pixels=sum(r.get('temporal_valid_pixels',0) for r in good))
 for k in ['RGB_MSE','coverage','soft_undercoverage','positive_excess','local_lpips','temporal_mse','foreground_psnr','full_psnr','ssim','crop_lpips','outside_alpha_mean']:
  if any(k in r for r in rows):out[k]=avg(good,k, 'pixels' if k in ['RGB_MSE','coverage'] else 'reference_alpha_mass' if k in ['soft_undercoverage','positive_excess'] else 'temporal_valid_pixels' if k=='temporal_mse' else None)
 if out.get('temporal_mse') is not None:out['temporal_RMSE']=math.sqrt(out['temporal_mse']);out['temporal_valid_fraction']=out['temporal_valid_pixels']/max(out['pixels'],1)
 return out

def main():
 allrows=[];aggregates=[];displacements=[];checks=[];oldchecks=[]
 for out in sorted((D/'results').iterdir()):
  if not (out/'complete.json').exists():continue
  key=out.name;ck=json.loads((out/'complete.json').read_text());checks.append(ck)
  for kind in ['global','expanded','parts']:
   rows=load(out/f'{kind}.csv')
   for r in rows:r['kind']=kind
   allrows+=rows
   for scope in ['fixed_global','failure_window','last_frame']:
    n=61 if key.startswith('N06') else 121
    sr=[r for r in rows if (r['in_global'] if scope=='fixed_global' else r['in_window'] if scope=='failure_window' else r['frame']==n-1)]
    for roi in sorted(set(r.get('roi','global') for r in sr)):
     for mode in ['M00','M10','M01','M11']:
      subset=[r for r in sr if r['mode']==mode and r.get('roi','global')==roi]
      if subset:aggregates.append(dict(key=key,kind=kind,scope=scope,roi=roi,mode=mode,**aggregate(subset)))
   # compare reused protocol against old numbers for every available matching M11 observation.
   if kind!='global':
    old=load(OLD/'results'/f'{key}_32999'/f'{kind}.csv');idx={(int(r['view']),int(r['frame']),str(int(r['side'])) if kind=='expanded' else r['part']):r for r in old};errs=[]
    for r in rows:
     if r['mode']!='M11':continue
     roi=r['roi'].split('_')[-1] if kind=='expanded' else r['roi'];o=idx[int(r['view']),int(r['frame']),roi]
     for x,y in [('RGB_MSE','RGB_MSE' if kind=='expanded' else 'mse'),('soft_undercoverage','soft_undercoverage'),('positive_excess','positive_excess'),('coverage','coverage'),('temporal_mse','temporal_mse')]:
      if x in r and r[x] is not None and o.get(y) is not None:errs.append(abs(r[x]-o[y]))
    oldchecks.append(dict(key=key,kind=kind,compared_values=len(errs),max_abs=max(errs,default=0.)))
  displacements+=load(out/'displacement.csv')
 write(D/'residual_modes.csv',allrows);write(D/'residual_summary.csv',aggregates);write(D/'displacement_support.csv',displacements)
 # Per-frame and aggregate conditional differences; paired NA never filled.
 interactions=[]
 for kind,collection in [('frame',allrows),('aggregate',aggregates)]:
  groups={}
  for r in collection:
   key=(r['key'],r['kind'],r.get('scope',''),r.get('view',''),r.get('frame',''),r.get('roi','global'));groups.setdefault(key,{})[r['mode']]=r
  for key,rs in groups.items():
   assert set(rs)=={'M00','M10','M01','M11'}
   for metric in ['RGB_MSE','soft_undercoverage','positive_excess','coverage','local_lpips','temporal_mse','temporal_RMSE','foreground_psnr','ssim','crop_lpips']:
    vals=[rs[m].get(metric) for m in ['M00','M10','M01','M11']]
    if not any(v is not None for v in vals):continue
    ok=all(v is not None for v in vals);e00,e10,e01,e11=vals
    interactions.append(dict(level=kind,key=key[0],kind=key[1],scope=key[2],view=key[3],frame=key[4],roi=key[5],metric=metric,direction='higher_better' if metric in ['coverage','foreground_psnr','ssim'] else 'lower_better',known=ok,M00=e00,M10=e10,M01=e01,M11=e11,position_given_color=e11-e01 if ok else None,color_given_position=e11-e10 if ok else None,position_without_color=e10-e00 if ok else None,color_without_position=e01-e00 if ok else None,interaction_J=e11-e10-e01+e00 if ok else None))
 write(D/'residual_interactions.csv',interactions)
 assert all(r['max_abs']<=1e-5 for r in oldchecks)
 (D/'research/metric_reproduction.json').write_text(json.dumps(oldchecks,indent=2))
 (D/'alpha_invariance.json').write_text(json.dumps(dict(tolerance=1e-5,passed=True,max_M11=max(r['M11_max'] for c in checks for r in c['checks']),max_alpha00_01=max(r['alpha00_01'] for c in checks for r in c['checks']),max_alpha10_11=max(r['alpha10_11'] for c in checks for r in c['checks']),observations=sum(len(c['checks']) for c in checks),per_asset=checks),indent=2))
 print('aggregates',len(aggregates),'interactions',len(interactions),'old metrics',oldchecks)
if __name__=='__main__':main()
