"""Observed stage costs and every registered quality/time point, no H-based selection."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
NEW=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def write(p,rows):
    with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
    rows=[];migration={r['label']:r for r in read(NEW/'research/fusion_migration.json')['jobs']};terminal={}
    for run in sorted((NEW/'runs').iterdir()):
        if not (run/'complete.json').exists():continue
        c=read(run/'complete.json');k=c.get('costs',{});native='counts' in c
        loop=c.get('training_wall_seconds') if native else k.get('training_loop_seconds',k.get('training_loop_wall_seconds',k.get('loop_seconds')))
        renders=c.get('counts',{}).get('training_renders') if native else k.get('render_calls',k.get('training_render_calls'))
        updates=c.get('counts',{}).get('optimizer_calls') if native else k.get('updates',k.get('gaussian_optimizer_calls',k.get('color_updates')))
        redundant=0.;discarded=0
        if run.name.endswith('_fused'):
            m=migration[run.name[:3]];redundant=m['last_record']['wall_training_seconds']-m['saved_training_wall_seconds'];discarded=m['discarded_updates']
        row=dict(job=run.name,category='training',logical_loop_seconds=loop,recorded_discarded_loop_seconds=redundant,recorded_discarded_updates=discarded,entry_end_to_end_seconds=c.get('end_to_end_seconds'),entry_scope='resumed entry only; prior logical loop and discarded work separately recorded' if run.name.endswith('_fused') else 'whole process entry',renders_logical=renders,backwards_logical=renders if native and not run.name.endswith('_fused') else c.get('counts',{}).get('optimizer_calls',0)+1 if native else k.get('backward_calls',k.get('training_backward_calls')),optimizer_updates=updates,statistics_seconds=k.get('statistics_seconds'),solve_seconds=k.get('solve_seconds'),fixed_pool_evaluation_seconds=k.get('full_pool_evaluation_seconds'),snapshot_io_seconds=k.get('snapshot_seconds',k.get('snapshot_and_recovery_seconds')),peak_allocated_bytes=c.get('peak_allocated_bytes'),peak_reserved_bytes=c.get('peak_reserved_bytes'),gpu_uuid=c['device']['uuid'])
        rows.append(row);terminal[run.name]=row
    for folder in sorted((NEW/'results').iterdir()):
        if not (folder/'metrics.json').exists():continue
        m=read(folder/'metrics.json');row={k:None for k in rows[0]};row.update(job=folder.name,category='global_and_optional_dense_evaluation',logical_loop_seconds=m['seconds'],optimizer_updates=0,gpu_uuid=m['device']['uuid']);rows.append(row)
    for p in sorted((NEW/'runs').glob('*/validation/*.json')):
        m=read(p);row={k:None for k in rows[0]}
        if p.stem.endswith('_timing'):
            row.update(job=p.parents[1].name+'_'+p.stem,category='additional_full_sequence_timed_replay',logical_loop_seconds=m['replay_wall_seconds'],entry_end_to_end_seconds=m['process_entry_seconds'],optimizer_updates=0,renders_logical=m['render_calls'],gpu_uuid=m['device']['uuid'])
        else:row.update(job=p.parents[1].name+'_'+p.stem,category='independent_replay_validation',logical_loop_seconds=m['seconds'],optimizer_updates=0,renders_logical=72,gpu_uuid=m['device']['uuid'])
        rows.append(row)
    write(NEW/'costs.csv',rows);curve=[]
    for label in ['N01','N07']:
        native=terminal[label+'_33k']['logical_loop_seconds'];alpha=terminal[label+'_alpha']['logical_loop_seconds']
        for stage,epochs in [('33k',[5000,15000,25000,33000]),('alpha',[38000,43000,48000]),('A',[49000,51000,53000]),('fused',[15000,20000,25000,33000])]:
            for epoch in epochs:
                key=f'{label}_{stage}_{epoch}';mp=NEW/'results'/key/'metrics.json';bp=NEW/'runs'/f'{label}_{stage}'/'budget'/f'{epoch}.json'
                if not mp.exists() or not bp.exists():continue
                m,b=read(mp),read(bp)
                if stage=='33k':seconds=b['training_loop_wall_so_far']
                elif stage=='fused':seconds=b['loop_seconds']
                else:seconds=native+(alpha if stage=='A' else 0)+b['loop_costs']['training_loop_seconds']
                curve.append(dict(label=label,stage=stage,actual_updates=m['actual_optimizer_updates'],logical_loop_seconds=seconds,**m['mean']))
    write(NEW/'efficiency_curve.csv',curve)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for row,label in enumerate(['N01','N07']):
        for stage,color in [('33k','#4c78a8'),('alpha','#f58518'),('A','#54a24b'),('fused','#b279a2')]:
            points=[r for r in curve if r['label']==label and r['stage']==stage]
            if not points:continue
            axes[row,0].plot([r['actual_updates']/1000 for r in points],[r['foreground_psnr'] for r in points],'-o',label=stage,color=color)
            axes[row,1].plot([r['logical_loop_seconds']/60 for r in points],[r['crop_lpips'] for r in points],'-o',label=stage,color=color)
        axes[row,0].set(title=f'{label}: every registered checkpoint',xlabel='Actual optimizer updates (thousands)',ylabel='Foreground PSNR (dB), higher better')
        axes[row,1].set(title=f'{label}: observed loop cost (shared-load caveat)',xlabel='Logical trajectory loop minutes',ylabel='Crop LPIPS, lower better')
        for ax in axes[row]:ax.grid(alpha=.2);ax.legend()
    fig.savefig(NEW/'comparison/quality_time.png',dpi=160);plt.close(fig)
    totals=[]
    for label in ['N01','N07']:
        names=[label+'_33k',label+'_alpha',label+'_A'];a=sum(terminal[n]['logical_loop_seconds'] for n in names);end=sum(terminal[n]['entry_end_to_end_seconds'] for n in names)
        row=dict(label=label,A_actual_updates=52999,A_loop_seconds=a,A_stage_entries_seconds=end,fused_actual_updates=None,fused_logical_loop_seconds=None,fused_recorded_discarded_seconds=None,fused_observed_loop_ratio=None)
        if label+'_fused' in terminal:
            f=terminal[label+'_fused'];row.update(fused_actual_updates=32999,fused_logical_loop_seconds=f['logical_loop_seconds'],fused_recorded_discarded_seconds=f['recorded_discarded_loop_seconds'],fused_observed_loop_ratio=(f['logical_loop_seconds']+f['recorded_discarded_loop_seconds'])/a)
        totals.append(row)
    save(NEW/'research/cost_accounting.json',dict(totals=totals,iteration_reduction_fraction=20000/52999,time_comparability='A ran mostly on free GPU6/7. Fused began on shared2/3 then migrated to6/7; observed wall ratios are not controlled algorithm speedups. GPU model/UUID in per-run start and resume evidence.',migration_cost='Discarded executed updates retained separately; in-flight work at interrupt not fully measurable. Native logical costs restore from last complete checkpoint.',scope='Backend only; frozen4DAnyone generation excluded. Color cost includes statistics,solver,full-pool reevaluation and snapshots separately. CUDA timeline is not pure GPU busy time.',unmeasured=['Original first replay video process and CPU part-evaluation wall time were not separately instrumented; reported NA, not zero.','Data hashing, checks and human visual inspection total wall cost not comprehensively isolated.'],quality_rule='Every registered checkpoint shown; no best checkpoint or stopping target selected using H.'))
    print('Cost and fixed checkpoint curves written',len(curve),flush=True)
if __name__=='__main__':main()
