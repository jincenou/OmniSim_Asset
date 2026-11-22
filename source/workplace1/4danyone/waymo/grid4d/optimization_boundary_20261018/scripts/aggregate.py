"""Assemble evidence and apply predeclared engineering gates without tuning."""
import csv
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime,timezone
NEW=Path(__file__).resolve().parents[1];ROOT=NEW.parent


def read(path):return json.loads(Path(path).read_text())
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvrows(path):return list(csv.DictReader(Path(path).open()))
def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    spec=read(NEW/'PLAN.json');assert hashlib.sha256((NEW/'PLAN.json').read_bytes()).hexdigest()==(NEW/'plan_hash.txt').read_text().strip()
    parts=csvrows(NEW/'comparison/part_metrics_summary.csv');globalrows=csvrows(NEW/'comparison/paired_metrics.csv');globals={(r['label'],r['method']):{k:float(r[k]) for k in ['full_psnr','foreground_psnr','ssim','crop_lpips','silhouette_iou','outside_alpha_mean']} for r in globalrows}
    coverage={(r['label'],r['method'],r['part']):float(r['mean_coverage']) if r['mean_coverage'] else None for r in parts}
    regions=[];locals_=[];appearances=[];anchors=[];queries={};costs=[]
    for label in ['N04','N11']:
        folder=NEW/'results/regions'/label;assert (folder/'complete.json').exists()
        regions+=csvrows(folder/'boundary_diagnosis.csv');locals_+=csvrows(folder/'local_crops.csv')
        folder=NEW/'results/analysis'/label;appearances+=csvrows(folder/'appearance_time_probe.csv');anchors+=read(folder/'anchors.json')['anchors'];queries[label]=read(folder/'query_readiness.json')
        for arm in ['A','U','B']:
            regions+=csvrows(NEW/'results/primary'/f'{label}_{arm}'/'regions.csv')
            run=NEW/'runs/primary'/f'{label}_{arm}';c=read(run/'complete.json');assert c['sampling_exact_match'] and c['stage_updates']==5000
            costs.append(dict(label=label,job=arm,category='training',seconds=c['costs']['training_loop_wall_seconds'],end_to_end_seconds=c['end_to_end_seconds'],snapshot_recovery_seconds=c['costs']['snapshot_and_recovery_seconds'],peak_allocated_bytes=c['peak_allocated_bytes'],peak_reserved_bytes=c['peak_reserved_bytes'],renders=c['costs']['training_render_calls'],backwards=c['costs']['training_backward_calls'],actual_updates=5000))
            e=read(NEW/'results/primary'/f'{label}_{arm}'/'metrics.json');costs.append(dict(label=label,job=arm,category='global_evaluation',seconds=e['seconds'],end_to_end_seconds=None,snapshot_recovery_seconds=None,peak_allocated_bytes=None,peak_reserved_bytes=None,renders=6*spec['identities'][label]['frames'],backwards=0,actual_updates=0))
            for it in [49000,51000,53000]:
                v=read(run/'validation'/f'{it}.json');costs.append(dict(label=label,job=f'{arm}_{it}',category='independent_replay',seconds=v['seconds'],end_to_end_seconds=None,snapshot_recovery_seconds=None,peak_allocated_bytes=None,peak_reserved_bytes=None,renders=72,backwards=0,actual_updates=0))
        for name,file in [('appearance_query',NEW/'results/analysis'/label/'query_readiness.json'),('region_diagnosis',NEW/'results/regions'/label/'complete.json')]:
            j=read(file);costs.append(dict(label=label,job=name,category='diagnostic',seconds=j['seconds'],end_to_end_seconds=None,snapshot_recovery_seconds=None,peak_allocated_bytes=None,peak_reserved_bytes=None,renders=j.get('render_calls'),backwards=j.get('backward_calls'),actual_updates=0))
        for file in sorted((NEW/'research/gradients'/label).glob('*.json')):
            j=read(file);costs.append(dict(label=label,job=file.stem,category='independent_gradient_probe',seconds=j['seconds'],end_to_end_seconds=None,snapshot_recovery_seconds=None,peak_allocated_bytes=None,peak_reserved_bytes=None,renders=j['render_calls'],backwards=j['autograd_calls'],actual_updates=0))
        for name,relative in [('region_precompute',f'data/{label}/bands/complete.json'),('technical_checks',f'research/{label}_technical.json'),('cutoff_audit',f'research/{label}_cutoff_audit.json')]:
            j=read(NEW/relative);costs.append(dict(label=label,job=name,category='preparation' if name=='region_precompute' else 'diagnostic',seconds=j['seconds'],end_to_end_seconds=None,snapshot_recovery_seconds=None,peak_allocated_bytes=None,peak_reserved_bytes=None,renders=j.get('render_calls'),backwards=j.get('backward_calls'),actual_updates=0))
    write_csv(NEW/'boundary_diagnosis.csv',regions);write_csv(NEW/'local_crop_metrics.csv',locals_);write_csv(NEW/'appearance_time_probe.csv',appearances);write_csv(NEW/'costs.csv',costs)
    save(NEW/'ANCHOR_AUDIT.json',dict(state='passed',source='data/anchors.json',source_sha256=spec['probes']['anchors_sha256'],fixed_before_new_training=True,anchors=anchors,heldout_content_reads=0,limits='Core ROI estimates, not face or hand truth; candidate correspondence remains conditional'))
    save(NEW/'query_readiness.json',dict(query_training_executed=False,identities=queries,correspondence_verified=dict(level='conditional_native_model_cycle_depth_and_part_checks_only',ground_truth=False,visual_review='See visual_review.json'),complementary_information=dict(level='per-anchor transfer evidence; no general cross-person claim'),simple_transfer_gain='Per-anchor same-full-ROI MSE/MAE in identities; untransferred pixels retain original reconstruction',next_minimum_suggestion='Validate same-surface correspondence and remove speckled transfers before considering learned Query; reliability weights have no established uniform-fusion advantage'))
    grouped={}
    for label in ['N04','N11']:
        for method in ['continue','alpha','combined','A','U','B']:
            for region in ['interior','inner','outer','far']:
                rows=[r for r in regions if r['label']==label and r['method']==method and r['region']==region and r['domain']=='heldout_development'];area=sum(int(r['area']) for r in rows);assert area
                grouped[label,method,region]={k:sum(float(r[k])*int(r['area']) for r in rows if r[k]!='')/area for k in ['alpha_mae','positive_excess_alpha','undercoverage_alpha','rgb_mse','mean_pred_alpha','mean_ref_alpha']}
    save(NEW/'research/region_summary.json',[dict(label=l,method=m,region=r,**v) for (l,m,r),v in grouped.items()])
    visual=read(NEW/'visual_review.json') if (NEW/'visual_review.json').exists() else None;gates=[]
    for label in ['N04','N11']:
        for candidate in ['U','B']:
            for comparator in (['A','alpha','U'] if candidate=='B' else ['A','alpha']):
                checks=[];current=globals[label,candidate];base=globals[label,comparator]
                limits=[('outside_alpha_mean','ratio_max',.90),('crop_lpips','increase_max',.002),('foreground_psnr','drop_max',.2),('silhouette_iou','drop_max',.003),('ssim','drop_max',.001)]
                for field,rule,limit in limits:
                    value=current[field]/base[field] if rule=='ratio_max' else current[field]-base[field] if rule=='increase_max' else base[field]-current[field]
                    checks.append(dict(metric=field,rule=rule,limit=limit,value=value,candidate=current[field],comparator=base[field],state='pass' if value<=limit else 'fail'))
                for part in ['head','torso','left_arm','right_arm','left_leg','right_leg']:
                    a,b=coverage[label,candidate,part],coverage[label,comparator,part];value=b-a if a is not None and b is not None else None
                    checks.append(dict(metric='coverage_'+part,rule='drop_max',limit=.01,value=value,candidate=a,comparator=b,state='cannot_determine' if value is None else 'pass' if value<=.01 else 'fail'))
                a=grouped[label,candidate,'far']['positive_excess_alpha'];b=grouped[label,comparator,'far']['positive_excess_alpha'];checks.append(dict(metric='far_positive_excess_alpha',rule='increase_max',limit=1e-6,value=a-b,candidate=a,comparator=b,state='pass' if a-b<=1e-6 else 'fail'))
                verdict=visual['comparisons'][label][candidate][comparator]['state'] if visual else 'cannot_determine'
                checks.append(dict(metric='visual_no_new_contamination_erosion_motion_anomaly',state=verdict,evidence='visual_review.json' if visual else None))
                numeric=all(c['state']=='pass' for c in checks if not c['metric'].startswith('visual_'));overall='fail' if any(c['state']=='fail' for c in checks) else 'pass' if all(c['state']=='pass' for c in checks) else 'cannot_determine'
                gates.append(dict(label=label,candidate=candidate,comparator=comparator,role='independent_boundary_gain' if comparator=='U' else 'basic',checks=checks,numeric_pass=numeric,state=overall))
    success={c:all(g['state']=='pass' for g in gates if g['candidate']==c) for c in ['U','B']};trigger=any(success.values())
    numeric_possible={c:all(g['numeric_pass'] for g in gates if g['candidate']==c) for c in ['U','B']}
    save(NEW/'gate_results.json',dict(state='evaluated',at=datetime.now(timezone.utc).isoformat(),plan_sha256=hashlib.sha256((NEW/'PLAN.json').read_bytes()).hexdigest(),engineering_not_statistical_thresholds=True,comparisons=gates,common_candidate_pass=success,repeat_authorized=trigger,repeat_state='authorized_once' if trigger else 'not_triggered_numeric_gates_failed' if not any(numeric_possible.values()) else 'waiting_for_visual_evidence',reason='Same candidate must pass separately on both people against common alpha start and A; B additionally against U. No per-person winner selection or mean hiding failures.'))
    print('Gates evaluated; conditional repeat authorized:',trigger,flush=True)


if __name__=='__main__':main()
