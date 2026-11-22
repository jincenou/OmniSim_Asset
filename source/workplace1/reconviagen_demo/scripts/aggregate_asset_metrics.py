"""AOR counts all frozen queries; appearance scores use one common valid subset."""
from pathlib import Path
import json,csv,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,default=ROOT/'inputs/random10_cases.json');p.add_argument('--output',default='random10_metrics');a=p.parse_args()
 cases=[c['case_id'] for c in json.loads(a.cases.read_text())];variants=['B0','R','F','H','B0_real1','B0_real4'];rows=[]
 for c in cases:
  for v in variants:
   path=ROOT/'results'/c/f'{v}_seed42'/'evaluation';render=path/'render_audit.json';row=dict(case_id=c,variant=v,valid=False,validation_state='pending',CLIP_I16=None,DINO_I16=None,ED_R=None,failure=None)
   if render.exists():
    audit=json.loads(render.read_text());row['valid']=audit['valid_export'] and audit['valid_16_views'];row['validation_state']='valid' if row['valid'] else 'failed'
   elif (path.parent/'failure.txt').exists():row['validation_state']='failed';row['failure']='Recorded generation failure'
   else:row['failure']='Pending generation or export/16-view validation; not counted as a completed failure'
   if row['valid']:
    for encoder,col in [('clip','CLIP_I16'),('dino','DINO_I16')]:
     p=path/f'{encoder}_metrics.json'
     if p.exists():
      d=json.loads(p.read_text());row[col]=d['I16']
      if encoder=='dino':row['ED_R']=d['ED_R']
   rows.append(row)
 common=[c for c in cases if all(r['valid'] for r in rows if r['case_id']==c)]
 stats=[]
 for v in variants:
  subset=[r for r in rows if r['variant']==v and r['case_id'] in common];summary=dict(variant=v,Nquery=len(cases),Nvalid=sum(r['valid'] for r in rows if r['variant']==v),Ncommon=len(common))
  summary['Npending']=sum(r['validation_state']=='pending' for r in rows if r['variant']==v)
  summary['AOR_percent']=100*summary['Nvalid']/len(cases) if summary['Npending']==0 else None
  for metric in ['CLIP_I16','DINO_I16','ED_R']:
   values=[r[metric] for r in subset]
   if not values or any(x is None for x in values):summary[metric]=None;continue
   x=np.asarray(values);rng=np.random.default_rng(20261023);means=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
   summary[metric]=dict(mean=float(x.mean()),sample_std=float(x.std(ddof=1)) if len(x)>1 else 0,bootstrap_95_percent_ci=np.quantile(means,[.025,.975]).tolist(),N=len(x))
  stats.append(summary)
 paired=[]
 for v in variants[1:]:
  delta_summary=dict(variant=v,reference='B0',N=len(common))
  for metric in ['CLIP_I16','DINO_I16','ED_R']:
   deltas=[]
   for c in common:
    target=next(r for r in rows if r['case_id']==c and r['variant']==v)[metric];base=next(r for r in rows if r['case_id']==c and r['variant']=='B0')[metric]
    if target is None or base is None:break
    deltas.append(target-base)
   if not common or len(deltas)!=len(common):delta_summary[metric]=None;continue
   x=np.asarray(deltas);rng=np.random.default_rng(20261023);means=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
   delta_summary[metric]=dict(mean_delta=float(x.mean()),bootstrap_95_percent_ci=np.quantile(means,[.025,.975]).tolist(),wins=int((x<0).sum()) if metric=='ED_R' else int((x>0).sum()))
  paired.append(delta_summary)
 result=dict(scope=f'Local {len(cases)}-identity Waymo pilot; input selection and implementation recorded; not paper-exact benchmark reproduction',common_valid_ids=common,rows=rows,summary=stats,missing_metrics_are_not_zero=True,validation_complete=all(r['validation_state']!='pending' for r in rows),metrics_complete=bool(common) and all(r[k] is not None for r in rows if r['case_id'] in common for k in ['CLIP_I16','DINO_I16','ED_R']))
 result['paired_differences_from_B0']=paired
 out=ROOT/'docs'/a.output;out.with_suffix('.json').write_text(json.dumps(result,indent=2))
 with out.with_suffix('.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
