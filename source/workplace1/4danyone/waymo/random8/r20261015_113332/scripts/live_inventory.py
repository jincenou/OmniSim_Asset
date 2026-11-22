"""Observe live processes separately from historical receipts; never kill processes."""
from pathlib import Path
import json,os,datetime
R=Path(__file__).resolve().parents[1]
def alive(pid):
 path=Path(f'/proc/{pid}/cmdline')
 try:return str(R).encode() in path.read_bytes()
 except (FileNotFoundError,ProcessLookupError,PermissionError):return False
def main():
 status=json.loads((R/'STATUS.json').read_text());rows=[]
 for key,job in status['jobs'].items():
  state=job['state'];pid=job.get('pid');is_live=bool(pid and alive(pid));row={'job':key,'receipt_state':state,'state':state,'pid':pid,'observed_process_alive':is_live,'log':job.get('log')}
  if state=='running' and not is_live:
   log=Path(job['log']).read_text(errors='replace') if job.get('log') else ''
   row['state']='failed_log_observed_exit_code_unknown' if 'Traceback (most recent call last)' in log else 'finished_artifact_requires_receipt_reconciliation'
  person=status['samples'].get(key.split('/')[0],{})
  if state.startswith('preregistered_') and person.get('state','').startswith('frontend_'):row['state']='not_run_frontend_failure'
  repair=R/'research'/(key.replace('/','_').replace('_evaluation','')+'_evaluation_repair.json')
  if key.endswith('_evaluation') and repair.exists() and json.loads(repair.read_text()).get('exit_code')==0:row['state']='complete_after_recorded_evaluation_retry'
  rows.append(row)
 repairs={p.stem:json.loads(p.read_text()) for p in (R/'research').glob('*evaluation_repair.json')};successful=[];progress=[]
 for p in (R/'samples').glob('*/evaluations/*/results.json'):
  result=json.loads(p.read_text());successful.append({'label':result['label'],'branch':result['branch'],'snapshot':result.get('snapshot'),'mean':result['mean'],'result':str(p)})
 for p in list((R/'samples').glob('*/ftgs_*/progress.json'))+list((R/'four_dg/runs').glob('*/progress.json')):
  v=json.loads(p.read_text());v['file']=str(p);v['observed_process_alive']=alive(v['pid']);progress.append(v)
 out={'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller_pid':status['controller_pid'],'controller_alive':alive(status['controller_pid']),'controller_state':status['state'],'jobs':rows,'progress':progress,'evaluation_repairs':repairs,'successful_evaluations':successful,'warning':'Historical running receipts do not prove live processes; finished progress is retained and explicitly distinguished.'};path=R/'LIVE.json';tmp=path.with_suffix('.writing');tmp.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');tmp.replace(path)
 branches=['ftgs_base5k','ftgs_opt15k','4dgs_grid25','4dgs_grid61'];matrix=[x for x in rows if x['job'].split('/')[-1] in branches];m=R/'EXPERIMENT_MATRIX.json';t=m.with_suffix('.writing');t.write_text(json.dumps({'target_new_tracks':8,'target_final_versions':32,'maximum_preregistered_tracks':16,'rows':matrix,'note':'Rows for inactive reserves are potential replacements, not extra target people. Skipped frontend failures remain in cohort denominator.'},indent=2)+'\n');t.replace(m);return out
if __name__=='__main__':main()
