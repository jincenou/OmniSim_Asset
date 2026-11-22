"""Postprocess only after authoritative training controller exits successfully."""
import asyncio,json,os,sys,time
from pathlib import Path
NEW=Path(__file__).resolve().parents[1]
state=dict(state='waiting_for_owned_controller',pid=os.getpid(),steps=[])
def save():
    p=NEW/'ANALYSIS_STATUS.json';q=p.with_suffix('.writing');q.write_text(json.dumps(state,indent=2)+'\n');q.replace(p)
async def main():
    save()
    while True:
        d=json.loads((NEW/'STATUS.json').read_text())
        if d['state']!='running':break
        # Host process namespace: missing confirmed controller is terminal failure, not an observation timeout.
        try:os.kill(d['pid'],0)
        except ProcessLookupError:
            state.update(state='failed',reason='Owned controller PID missing while status stillrunning; inspect logs, no automatic restart');save();return
        await asyncio.sleep(30)
    if d['state']!='training_evaluation_complete_review_pending':
        state.update(state='failed',reason='Controller finished with recorded failure; no fabricated partial completion');save();return
    state['state']='running_audit_and_analysis';save()
    for script in ['audit_cross8.py','analyze_cross8.py','summarize_costs.py']:
        step=dict(script=script,started_at=time.time());state['steps'].append(step);save()
        with (NEW/'logs'/('final_'+script+'.log')).open('x') as log:
            p=await asyncio.create_subprocess_exec(sys.executable,str(NEW/'scripts'/script),stdout=log,stderr=asyncio.subprocess.STDOUT);step['pid']=p.pid;save();step['exit_code']=await p.wait()
        step['ended_at']=time.time();save()
        if step['exit_code']:
            state.update(state='failed',reason='Postprocess script failed; preserve logs');save();return
    state['state']='numeric_and_asset_audit_complete_visual_report_pending';save()
if __name__=='__main__':asyncio.run(main())
