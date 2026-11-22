"""Switch schedulers only after the current native training and evaluation exit.

No training process is interrupted or restarted. Separate first single-card
validation and frontend repairs keep running across this CPU-only handoff.
"""
import asyncio,json,os,signal,datetime
from pathlib import Path
from live_inventory import alive
R=Path(__file__).resolve().parents[1]
def write(v):
 p=R/'CONTROLLER_HANDOFF.json';t=p.with_suffix('.writing');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
async def main():
 while True:
  state=json.loads((R/'STATUS.json').read_text());pid=state['controller_pid'];cmd=Path(f'/proc/{pid}/cmdline')
  if cmd.exists() and b'controller_v2.py' in cmd.read_bytes():write({'state':'v2_already_live','pid':pid});return
  train=R/'logs/N04_4dgs_grid25.json';evaluation=R/'logs/N04_4dgs_grid25_evaluation.json'
  trained=train.exists() and json.loads(train.read_text()).get('exit_code')==0
  evaluated=evaluation.exists() and json.loads(evaluation.read_text()).get('state') in ['complete','failed']
  live=[(k,v.get('pid')) for k,v in state['jobs'].items() if v.get('state')=='running' and v.get('pid') and alive(v['pid'])]
  write({'state':'waiting_current_native_training_and_eval_to_exit','controller_pid':pid,'trained':trained,'evaluation_terminal':evaluated,'live_owned_jobs':live,'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
  if trained and evaluated and not live:
   assert cmd.exists() and b'/scripts/controller.py' in cmd.read_bytes();snapshot=R/'research/controller_v1_terminal_snapshot.json';snapshot.write_text(json.dumps(state,indent=2)+'\n');os.kill(pid,signal.SIGTERM)
   for _ in range(20):
    if not cmd.exists():break
    await asyncio.sleep(.25)
   if cmd.exists() and cmd.read_bytes():raise RuntimeError('Oldcontroller has not exited;do not start anotherwriter')
   write({'state':'starting_v2_after_verified_idle_handoff','old_controller_pid':pid,'snapshot':str(snapshot),'no_gpu_job_interruption':True})
   # Retire only the old companion accounting loop; v2 owns its own terminal accounting.
   p=await asyncio.create_subprocess_exec('tmux','-L','fdwaymo','kill-window','-t','random8_r20261015_113332:final-accounting');await p.wait()
   os.execv('/mnt/ojc/miniconda3/envs/4danyone/bin/python',['/mnt/ojc/miniconda3/envs/4danyone/bin/python','-u',str(R/'scripts/controller_v2.py')])
  if not alive(pid):write({'state':'blocked_original_controller_missing_requires_review','pid':pid});return
  await asyncio.sleep(30)
if __name__=='__main__':asyncio.run(main())
