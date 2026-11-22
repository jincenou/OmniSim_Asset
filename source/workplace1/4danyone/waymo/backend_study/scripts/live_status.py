"""Compact eight-case status derived from receipts, outputs and live process identity."""
import asyncio,json,os
from pathlib import Path
from run_queue import ROOT,save,stamp
LABELS=['N01','N02','N03','N04','N06','N07','N08','N11']

def snapshot():
    rows=[]
    for label in LABELS:
        row={'label':label,'state':'not_started'}
        for stage in ['stage1','stage2','hust_raw','evaluate']:
            receipt=ROOT/'logs'/f'{label}_{stage}.json'
            if not receipt.exists():continue
            value=json.loads(receipt.read_text());row.update(stage=stage,state=value['state'],pid=value.get('pid'),gpu=value['gpu']['physical_gpu'])
            if value['state']=='running':
                proc=Path('/proc')/str(value['pid']);live=False
                if (proc/'cmdline').exists():
                    cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
                    live=str(ROOT/'scripts') in cmd and label in cmd
                row['process_live']=live
                if not live:row['state']='receipt_exit_update_pending'
            progress=ROOT/'runs'/label/stage/'progress.json'
            if progress.exists():
                p=json.loads(progress.read_text());row.update(iteration=p['iteration'],motion_groups=p['motion_groups'],points=p['points'],progress_at=p['updated_at'])
        result=ROOT/'results'/label/'results.json'
        if result.exists():row.update(state='evaluated_visual_review_pending',result=str(result))
        rows.append(row)
    done=sum(x['state']=='evaluated_visual_review_pending' for x in rows)
    terminal=all(x['state'] in ['failed','evaluated_visual_review_pending'] for x in rows)
    record={'updated_at':stamp(),'target_cases':8,'evaluated_cases':done,
        'state':'evaluated_visual_review_pending' if done==8 else ('failed_partial' if terminal else 'in_progress'),'cases':rows,
        'primary_walking':['N04','N08'],'separate_stress_case':'N11',
        'main_cohort':['N01','N02','N03','N04','N06','N07','N08'],'production_approved':False}
    save(ROOT/'LIVE_STATUS.json',record)
    return terminal

async def main():
    while not snapshot():await asyncio.sleep(60)

if __name__=='__main__':asyncio.run(main())
