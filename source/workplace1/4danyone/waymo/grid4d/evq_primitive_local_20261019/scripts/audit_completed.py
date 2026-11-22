"""Read-only recovery/hash/optimizer audit of all24 required snapshots."""
import json,gc,hashlib
from pathlib import Path
import torch
N=Path(__file__).resolve().parents[1];OLD=N.parent/'early_visual_query_20261019'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 t=p.with_suffix('.writing');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def main():
 torch.set_num_threads(2);plan=read(N/'PLAN.json');records=[]
 for p,h in plan['implementation_sha256'].items():assert sha(N/p)==h,p
 for p,h in plan['evaluation']['expanded_ROI_sha256'].items():assert sha(N/p)==h,p
 for label in plan['labels']:
  schedule=read(N/'data'/label/'schedule.json');warm=OLD/'runs'/f'{label}_warmup/recovery_2999.pt';warmhash=sha(warm)
  for arm in plan['arms']:
   key=f'{label}_{arm}';run=N/'runs'/key;resume=read(run/'resume.json');assert resume['sha256']==warmhash and Path(resume['source'])==warm and resume['step']==2999
   trace=[json.loads(s) for s in (run/'trace.jsonl').read_text().splitlines()];assert trace[-1]['step']==32999
   for row in trace:assert [row['frame'],*row['views']]==schedule[row['step']-1]
   done=read(run/'training.complete.json');assert done['counts']['additional_updates']==30000
   snapshots=[];start=None
   for step in [5000,15000,18000,32999]:
    p=run/f'recovery_{step}.pt';snap=read(run/f'snapshot_{step}.json');h=sha(p);assert h==snap['recovery_sha256'];s=torch.load(p,map_location='cpu');assert s['schema']=='evq_primitive_recovery_v1' and s['step']==step
    assert s['extra']['plan_sha256']==sha(N/'PLAN.json');c=s['extra']['counts'];assert c==snap['counts'];assert c['additional_updates']==step-2999 and c['target_renders']==2*(step-2999) and c['alpha_renders']==2*max(0,step-15000)
    assert all(c[k]==step-2999 for k in ['gaussian_optimizer_calls','query_optimizer_calls','deform_optimizer_calls'])
    n=len(s['gaussians']['_xyz']);assert n==snap['points']
    for t in s['gaussians'].values():assert len(t)==n and torch.isfinite(t).all()
    optsteps=sorted(set(int(x['step']) for x in s['query_optimizer']['state'].values()));assert optsteps==[step-2999],(key,step,optsteps)
    if step==5000:start={k:s['query'][k].clone() for k in ['query.weight','key.weight','value.weight','offset.weight','head.weight']}
    delta={k:float((s['query'][k]-v).abs().max()) for k,v in start.items()}
    snapshots.append(dict(step=step,sha256=h,points=n,bytes=p.stat().st_size,query_optimizer_steps=optsteps,gaussian_actual_updates=c['gaussian_groups_updated'],parameter_change_from5000=delta));del s;gc.collect()
   diag=[r['diagnostics'] for r in trace if r['diagnostics']];records.append(dict(key=key,passed=True,snapshots=snapshots,last_diagnostic=diag[-1],formal_gradient_max={k:max(r['query_grad_abs_sum'].get(k,0) for r in trace) for k in start},traced_observations_match_schedule=True,not_all_steps_individually_logged=True))
   save(N/'research/completed_training_audit.json',dict(complete=len(records)==6,runs=records));print('audited',key,flush=True)
if __name__=='__main__':main()
