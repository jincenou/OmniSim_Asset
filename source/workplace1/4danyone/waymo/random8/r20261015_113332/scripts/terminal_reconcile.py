"""Finish accounting and unused preregistered reserves after the primary queue exits."""
import asyncio,json,os
from pathlib import Path
from live_inventory import alive,main as observe
R=Path(__file__).resolve().parents[1]
async def main():
 while True:
  s=json.loads((R/'STATUS.json').read_text())
  if s['state'].startswith('all_registered_tasks_terminal') and not alive(s['controller_pid']):break
  if not alive(s['controller_pid']) and not s['state'].startswith('all_registered_tasks_terminal'):
   (R/'research/finalizer_blocked.json').write_text(json.dumps({'state':'blocked_controller_exit_needs_review','no_automatic_failed_job_restart':True},indent=2)+'\n');return
  await asyncio.sleep(120)
 import controller as c
 c.state.update(s,controller_pid=os.getpid(),state='final_accounting_and_ordered_unused_reserves')
 # Reconcile historical orphan receipts using artifacts/logs, never invent exit codes.
 for label,person in c.state['samples'].items():
  if person.get('state')=='frontend_failed_implementation_or_input_review_required':
   logs=[p for p in (R/'logs').glob(label+'_*attempt*.log')];text='\n'.join(p.read_text(errors='replace')[-4000:] for p in logs)
   if any(v in text for v in ['Global/incam SMPL geometry is not a consistent rigid pair','Exposure span shorter than fixed 25fps window','Annotated person clipped by crop','Common frontend foreground gate failed']):person.update(state='frontend_hard_failure',reserve_eligible=True,accounting_note='Observed completed failure log,including orphan child; exit code remains unavailable where not captured.')
 from disk_guard import main as guard
 guard_task=asyncio.create_task(guard())
 own=(R/'controller.lock').open('a')
 import fcntl
 fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);c.persist()
 def entered():return sum((R/'samples'/x['sample_id']/'data/init.npz').exists() for x in c.selection['main']+c.selection['reserve'])
 remaining=[x for x in c.selection['reserve'] if not (R/'samples'/x['sample_id']/'source').exists() and c.state['samples'].get(x['sample_id'],{}).get('state') in [None,'queued']]
 if any(v.get('reserve_eligible') for v in c.state['samples'].values()):
  while entered()<8 and remaining:
   person=remaining.pop(0);await c.person(person)
 # Clear success-vs-history ambiguity without deleting failed attempt receipts.
 for label,person in c.state['samples'].items():
  evaluated={b for b in ['ftgs_base5k','ftgs_opt15k','4dgs_grid25','4dgs_grid61'] if (R/'samples'/label/'evaluations'/b/'results.json').exists()}
  person['actual_evaluated_final_branches']=sorted(evaluated)
  if len(evaluated)==4:person.update(state='four_terminal_versions_evaluated_visual_review_pending')
 c.state.update(state='all_registered_tasks_terminal_accounted_review_pending',entered_backend_tracks=entered(),finalizer_pid=os.getpid());c.persist();observe()
 from report import main as report
 report()
 await guard_task
if __name__=='__main__':asyncio.run(main())
