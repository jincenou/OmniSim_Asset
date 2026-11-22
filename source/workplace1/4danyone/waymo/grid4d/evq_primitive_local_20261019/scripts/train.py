"""Six early branches only; exact old update ordering, new appearance read path."""
from update import *
OLD=ROOT/'early_visual_query_20261019'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--label',required=True,choices=['N06','N11']);ap.add_argument('--arm',required=True,choices=['G0','G1','G2']);ap.add_argument('--resume');a=ap.parse_args();plan=specification();assert plan['state']=='frozen_before_training';assert json.loads((NEW/'research/preflight.json').read_text())['passed']
 for p,h in plan['implementation_sha256'].items():assert sha(NEW/p)==h,p
 sys.addaudithook(guard_training);device=verify();run=NEW/'runs'/f'{a.label}_{a.arm}'
 if a.resume:assert run.exists()
 else:run.mkdir(exist_ok=False)
 started=time.monotonic();g,d,q,qo,cfg,data=cold(a.label,a.arm);q.chunk=plan['model']['local_chunk'];cfg['primitive_variant']=a.arm;cfg['evq_arm']='R3';cfg['dataset']['model_path']=str(run)
 source=Path(a.resume) if a.resume else OLD/'runs'/f'{a.label}_warmup/recovery_2999.pt';s=torch.load(source,map_location='cpu');step0=restore(s,g,d,q,qo,cfg,allow_cold=not a.resume)
 if a.resume:counts=s['extra']['counts'];elapsed_previous=s['extra']['training_wall_seconds']
 else:counts=dict(additional_updates=0,logical_terminal=2999,gaussian_optimizer_calls=0,deform_optimizer_calls=0,query_optimizer_calls=0,gaussian_groups_updated={},target_renders=0,alpha_renders=0);elapsed_previous=0.
 del s;inp=Inputs(a.label,data,True);schedule=json.loads((NEW/'data'/a.label/'schedule.json').read_text());loaded=time.monotonic();initial_rows=len(g.get_xyz);param_ids=[id(p) for group in qo.param_groups for p in group['params']];assert len(param_ids)==len(set(param_ids))
 save(run/'resume.json',dict(source=str(source),sha256=sha(source),step=step0,exact_restore=True,formal_new_parameters_seed=1042,shared_warmup=not bool(a.resume)));save(run/'config.json',cfg);save(run/'cameras.json',data);save(run/'device.json',device)
 save(run/'parameter_counts.json',dict(gaussian_at_start=sum(getattr(g,k).numel() for k in PARAMETERS),base_hash=sum(p.numel() for p in d.grid4d.parameters()),base_deform=sum(p.numel() for p in d.deform.parameters()),query=sum(p.numel() for p in q.parameters()),parameter_objects_unique=True,local_query_extra_parameters=0,geometry_and_appearance_share_parameters=True,local_chunk=q.chunk))
 trace=(run/'trace.jsonl').open('a' if a.resume else 'x',buffering=1)
 for step in range(step0+1,33000):
  if step%100==1:assert shutil.disk_usage(NEW).free>5*2**30,'5GiB disk reserve'
  frame,v1,v2=schedule[step-1];observe=step%100==0 or step in [3000,3001,3999,5000,15000,18000,32999];r=update(g,d,q,qo,cfg,data,inp,frame,[v1,v2],step,observe=observe)
  counts['additional_updates']+=1;counts['logical_terminal']=step;counts['gaussian_optimizer_calls']+=1;counts['deform_optimizer_calls']+=1;counts['query_optimizer_calls']+=1;counts['target_renders']+=2;counts['alpha_renders']+=2*int(step>15000)
  for k,value in r['gaussian_groups_updated'].items():counts['gaussian_groups_updated'][k]=counts['gaussian_groups_updated'].get(k,0)+int(value)
  if observe or r['structure']:
   row=dict(step=step,frame=frame,views=[v1,v2],wall_seconds=time.monotonic()-loaded,**r);trace.write(json.dumps(row,allow_nan=False)+'\n');save(run/'progress.json',dict(state='training',step=step,total=32999,additional_updates=counts['additional_updates'],points=len(g.get_xyz),device=device,loop_seconds=time.monotonic()-loaded,at=stamp()));print(a.label,a.arm,step,r['points'],r['loss'],flush=True)
  if step in plan['budget']['checkpoints'] or step%2000==0:
   extra=dict(counts=counts.copy(),training_wall_seconds=elapsed_previous+time.monotonic()-loaded,load_seconds=loaded-started,plan_sha256=sha(NEW/'PLAN.json'),schedule_sha256=sha(NEW/'data'/a.label/'schedule.json'),variant=a.arm,label=a.label,device=device,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
   target=run/(f'recovery_{step}.pt' if step in plan['budget']['checkpoints'] else 'recovery_latest.pt');atomic_torch(target,checkpoint(g,d,q,qo,cfg,data,step,extra))
   if step in plan['budget']['checkpoints']:save(run/f'snapshot_{step}.json',dict(step=step,points=len(g.get_xyz),active_sh_degree=g.active_sh_degree,recovery_sha256=sha(target),**extra))
 assert counts['additional_updates']==30000 and counts['logical_terminal']==32999;trace.close();save(run/'training.complete.json',dict(step=32999,counts=counts,seconds=time.monotonic()-started,load_seconds=loaded-started,loop_seconds=time.monotonic()-loaded,device=device,points=len(g.get_xyz)))
 # The required four full snapshots remain. Own rolling duplicate is dispensable after terminal hash exists.
 if (run/'recovery_latest.pt').exists():(run/'recovery_latest.pt').unlink()
 print(a.label,a.arm,'TRAIN COMPLETE',flush=True)
if __name__=='__main__':main()
