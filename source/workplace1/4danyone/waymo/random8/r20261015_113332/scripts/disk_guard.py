"""Protect the explicit 35GiB reserve; suspend only this run's own process groups."""
import asyncio,json,os,shutil,signal,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[1];STOP=35.75*1024**3;RESUME=36.5*1024**3

def own(pid):
 try:return str(R).encode() in Path(f'/proc/{pid}/cmdline').read_bytes() and os.getpgid(pid)==pid
 except (FileNotFoundError,ProcessLookupError,PermissionError):return False
async def main():
 paused={};path=R/'DISK_GUARD.json'
 while True:
  state=json.loads((R/'STATUS.json').read_text());free=shutil.disk_usage(R).free
  if free<STOP:
   jobs=[(k,v) for k,v in state['jobs'].items() if v['state']=='running' and v.get('physical_gpus')]
   jobs.extend((p.stem,json.loads(p.read_text())) for p in (R/'research').glob('*evaluation_repair.json'))
   repair_state=R/'FRONTEND_REPAIR_STATUS.json'
   if repair_state.exists():
    repair=json.loads(repair_state.read_text());jobs.extend((f'frontend-repair/{label}',v['job']) for label,v in repair['cases'].items() if v.get('job',{}).get('state')=='running' and v['job'].get('physical_gpus'))
   for key,v in jobs:
    pid=v.get('pid')
    if pid and own(pid) and pid not in paused:os.killpg(pid,signal.SIGSTOP);paused[pid]={'job':key,'pid':pid,'paused_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  if paused and free>=RESUME:
   for pid in list(paused):
    if own(pid):os.killpg(pid,signal.SIGCONT)
    del paused[pid]
  record={'state':'blocked_disk_own_gpu_processes_suspended' if paused else 'reserve_guard_active','free_bytes':free,'reserve_bytes':35*1024**3,'pause_threshold_bytes':int(STOP),'resume_threshold_bytes':int(RESUME),'paused_process_groups':list(paused.values()),'scope':'Only validated own random8 run process groups, never other projects. All pauses are recoverable without restarting models.','updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()};tmp=path.with_suffix('.writing');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(path)
  if state['state'].startswith('all_registered_tasks_terminal') and not paused:return
  await asyncio.sleep(2)
if __name__=='__main__':asyncio.run(main())
