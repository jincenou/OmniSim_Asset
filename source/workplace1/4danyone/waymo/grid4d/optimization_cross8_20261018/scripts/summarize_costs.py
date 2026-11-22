"""Observed process/loop costs with explicit historical and shared-load limitations."""
import csv,json
from pathlib import Path
NEW=Path(__file__).resolve().parents[1]
from publish_cross8 import descriptors,PLAN
def read(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write(p,rows):
    with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    status=read(NEW/'STATUS.json');records=[];summaries=[]
    for label in PLAN['new_A_labels']+['N04','N11']:
        names=[label+'_fused']
        if label in PLAN['new_A_labels']:names=[label+'_33k',label+'_alpha',label+'_A']+names
        entries={}
        for name in names:
            p=NEW/'runs'/name;c=read(p/'complete.json');cost=c.get('costs',{});native='counts' in c
            loop=c['training_wall_seconds'] if native else cost['training_loop_seconds']
            count=c['counts']['optimizer_calls'] if native else cost['updates']
            renders=c['counts']['training_renders'] if native else cost['render_calls']
            job=status['jobs'][label+('_fused' if name.endswith('_fused') else '_A')]
            stage_name=label+'_native_train' if name.endswith('_33k') else name+'_train'
            outer=next(s for s in job['stages'] if s['name']==stage_name)
            r=dict(run=name,category='training',loop_seconds=loop,recorded_entry_seconds=c['end_to_end_seconds'],full_subprocess_wall_seconds=outer['ended_at']-outer['started_at'],renders=renders,backwards=33000 if native else cost['backward_calls'],optimizer_updates=count,RGB_observations=33000 if native else count,RGB_pixels=(33000 if native else count)*352*640,snapshot_seconds=cost.get('snapshot_seconds'),peak_allocated_bytes=c.get('peak_allocated_bytes'),peak_reserved_bytes=c.get('peak_reserved_bytes'),gpu_uuid=c['device']['uuid'],gpu_index=job['gpu'],shared_load_caveat=True)
            if r['peak_reserved_bytes'] is None:r['peak_reserved_bytes']=read(p/'progress.json').get('peak_reserved_bytes')
            records.append(r);entries[name]=r
        if label in PLAN['new_A_labels']:
            a=[entries[label+'_'+s] for s in ['33k','alpha','A']];f=entries[label+'_fused']
            summaries.append(dict(label=label,A_actual_updates=52999,fused_actual_updates=32999,A_loop_seconds=sum(r['loop_seconds'] for r in a),fused_loop_seconds=f['loop_seconds'],A_full_subprocess_seconds=sum(r['full_subprocess_wall_seconds'] for r in a),fused_full_subprocess_seconds=f['full_subprocess_wall_seconds'],A_gpu=a[0]['gpu_index'],fused_gpu=f['gpu_index'],comparison='Observed wall under independently shared GPU loads; not controlled same-device algorithm speedup'))
    for job in status['jobs'].values():
        for stage in job['stages']:
            if stage['name'].endswith('_train'):continue
            r={k:None for k in records[0]};r.update(run=stage['name'],category='validation_or_evaluation',full_subprocess_wall_seconds=stage['ended_at']-stage['started_at'],optimizer_updates=0,gpu_uuid=job['uuid'],gpu_index=job['gpu'],shared_load_caveat=True);records.append(r)
    for path in sorted((NEW/'runs').glob('*/validation/*_timing.json')):
        d=read(path);r={k:None for k in records[0]};r.update(run=path.parents[1].name,category='full_sequence_asset_only_replay',loop_seconds=d['replay_wall_seconds'],recorded_entry_seconds=d['process_entry_seconds'],renders=d['render_calls'],optimizer_updates=0,gpu_uuid=d['device']['uuid'],shared_load_caveat=True);records.append(r)
    write(NEW/'costs.csv',records)
    save(NEW/'cost_summary.json',dict(state='measured_current_round',new_matched_pairs=summaries,update_reduction_fraction=20000/52999,reused_pairs='N01/N07 costs and fusion migration caveat in quality_v3; N04/N11 A stages span earlier runs, not fabricated as fresh matched timing.',scope='Backend only. Full subprocess wall includes interpreter/import/data/IO; recorded entry and loop use narrower instrumentation. Frontend generation, queue wait and final human review are excluded.',no_pure_GPU_busy_time_claim=True))
    print('Measured costs saved for new training and evaluation',flush=True)
if __name__=='__main__':main()
