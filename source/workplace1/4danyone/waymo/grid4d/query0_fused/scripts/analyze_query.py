"""Fixed protocol analysis: retain every person, arm, body core and unknown."""
import csv,hashlib,importlib.util,json,math
from pathlib import Path
import numpy as np
from runtime import NEW,ROOT,save
PLAN=json.loads((NEW/'PLAN.json').read_text());BASE=ROOT.parent/'random8/r20261015_113332'
spec=importlib.util.spec_from_file_location('frozen_aggregation',ROOT/'optimization_quality_v3_20261018/scripts/analyze_quality.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
PARTS=prior.PARTS;FIELDS=prior.FIELDS
def read(path):return json.loads(Path(path).read_text())
def csvrows(path):return list(csv.DictReader(Path(path).open()))
def write(path,rows):
    rows=list(rows);assert rows
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def descriptor(label,method):
    if method=='HUST':return dict(metrics=BASE/'samples'/label/'evaluations/4dgs_grid61/results.json',parts=ROOT/'audit/metrics'/label/'4dgs_grid61.csv',raw=ROOT/'tmp/baseline_raw'/label/'4dgs_grid61',run=None,actual=None)
    if method in ['F','alpha48','A']:
        arm={'F':'fused','alpha48':'alpha','A':'A'}[method];d=PLAN['anchors'][label][arm]
        return dict(metrics=Path(d['metrics']),parts=Path(d['parts']),raw=Path(d['raw']),run=Path(d['run']),actual={'F':32999,'alpha48':47999,'A':52999}[method],iteration=d['iteration'])
    anchor='A' if method.startswith('A+') else 'fused';arm=method[-1];actual=54999 if anchor=='A' else 34999;name=f'{label}_{anchor}_{arm}';result=NEW/'results'/f'{name}_{actual}'
    return dict(metrics=result/'metrics.json',parts=result/'parts.csv',raw=NEW/'cache/raw'/result.name,run=NEW/'runs'/name,actual=actual,iteration=actual)
def protection(label,candidate,comparator,metrics,parts,outside=True):
    a,b=metrics[label,candidate],metrics[label,comparator];checks=[]
    for field,rule,limit in [('foreground_psnr','drop_max',.2),('ssim','drop_max',.001),('crop_lpips','increase_max',.002)]:checks.append(prior.comparison_check(field,rule,limit,a[field],b[field]))
    for part in PARTS:
        p,q=parts[label,candidate][part],parts[label,comparator][part]
        checks.append(prior.comparison_check('coverage_'+part,'drop_max',.01,p['coverage'],q['coverage']))
        checks.append(prior.comparison_check('temporal_rmse_'+part,'ratio_max',1.05,p['compensated_rmse'],q['compensated_rmse']))
    if outside:checks.append(prior.comparison_check('outside_alpha_mean','ratio_max',1.05,a['outside_alpha_mean'],b['outside_alpha_mean']))
    return checks
def outcome(checks):
    failed=[c['metric'] for c in checks if c['state']=='fail'];unknown=[c['metric'] for c in checks if c['state']=='unknown']
    return dict(checks=checks,numeric_pass=not failed and not unknown,failed=failed,unknown=unknown,visual='not_established_by_numeric_results')
def worst_windows(label,method,rows):
    result=[]
    for view in [0,8,16]:
        for part in PARTS:
            selected=sorted([r for r in rows if int(r['view'])==view and r['part']==part],key=lambda r:int(r['frame']));windows=[]
            for start in range(len(selected)-24):
                seq=selected[start:start+25];valid=[r for r in seq if r['coverage']!=''];pixels=sum(int(r['reference_core_pixels']) for r in valid)
                if pixels:
                    windows.append(dict(first_frame=int(seq[0]['frame']),last_frame=int(seq[-1]['frame']),valid_slots=len(valid),reference_core_pixels=pixels,coverage=sum(float(r['coverage'])*int(r['reference_core_pixels']) for r in valid)/pixels,mse=sum(float(r['mse'])*int(r['reference_core_pixels']) for r in valid)/pixels))
            for name,key,reverse in [('lowest_coverage','coverage',False),('highest_mse','mse',True)]:
                w=sorted(windows,key=lambda r:r[key],reverse=reverse)[0] if windows else dict(first_frame=None,last_frame=None,valid_slots=0,reference_core_pixels=0,coverage=None,mse=None)
                result.append(dict(label=label,method=method,view=view,part=part,selection=name,**w,window_slots=25,total_candidate_windows=len(selected)-24,status='measured_with_reference_validity' if windows else 'unknown_no_reference_core'))
    return result
def main():
    metrics={};parts={};raw={};global_rows=[];summary=[];fullrows=[];worst=[];denominators=[]
    for label in PLAN['labels']:
        methods=['HUST','F','alpha48','A']+(['U','P','E','Q'] if label in PLAN['query_labels'] else [])
        signature=None
        for method in methods:
            d=descriptor(label,method);m=read(d['metrics']);rs=csvrows(d['parts']);metrics[label,method]=m['mean'];parts[label,method]=prior.part_summary(rs);raw[label,method]=rs
            sig=[(r['view'],r['frame'],r['part'],r['reference_core_pixels'],r['temporal_valid_pixels']) for r in rs]
            if signature is None:signature=sig
            else:assert sig==signature,(label,method,'Reference denominator mismatch')
            global_rows.append(dict(label=label,cohort='reserve' if label=='N11' else 'main',method=method,actual_updates=d['actual'],nominal_HUST_count=33000 if method=='HUST' else None,**{k:m['mean'][k] for k in FIELDS},metrics_file=str(d['metrics'])))
            for part in PARTS:summary.append(dict(label=label,method=method,part=part,**parts[label,method][part]))
            for r in rs:fullrows.append(dict(label=label,method=method,**{k:r[k] for k in ['view','frame','part','reference_core_pixels','coverage','soft_missing','mse','psnr','outside_reference_alpha','temporal_mse','temporal_valid_pixels','temporal_valid_fraction','status']}))
            worst.extend(worst_windows(label,method,rs))
        denominators.append(dict(label=label,state='passed',rows=len(signature),compared_methods=methods))
    write(NEW/'paired_metrics.csv',global_rows);write(NEW/'part_time_metrics.csv',fullrows);write(NEW/'part_summary.csv',summary);write(NEW/'worst25.csv',worst)
    alpha=[]
    for label in PLAN['labels']:alpha.append(dict(label=label,**outcome(protection(label,'alpha48','A',metrics,parts,outside=False))))
    save(NEW/'alpha48_gates.json',dict(state='all_eight_fixed_stages_evaluated',comparisons=alpha,old_protection_lines_unchanged=True,unknown_not_pass=True,actual_update_reduction=5000/52999,visual_review='visual_review.json when completed',no_individual_best_selection=True))
    write(NEW/'alpha48_evaluation.csv',[r for r in global_rows if r['method'] in ['alpha48','A']])
    comparisons=[];target_checks=[]
    for label in PLAN['query_labels']:
        for method in ['P','E','Q']:
            comparisons.append(dict(label=label,candidate=method,comparator='U',**outcome(protection(label,method,'U',metrics,parts))))
        for method in ['U','P','E','Q']:
            comparisons.append(dict(label=label,candidate=method,comparator='A',comparison_role='Old A replacement protection; no additional outsidealpha gate',**outcome(protection(label,method,'A',metrics,parts,outside=False))))
    for label in ['N06','N11']:
        targets=PLAN['gates']['target_parts'][label];u=[parts[label,'U'][part]['coverage'] for part in targets]
        weak=min(range(2),key=lambda i:u[i]) if all(x is not None for x in u) else None
        checks=[]
        for i,part in enumerate(targets):checks.append(prior.comparison_check('target_coverage_'+part,'drop_max',-.02 if i==weak else .01,parts[label,'Q'][part]['coverage'],parts[label,'U'][part]['coverage']))
        target_checks.append(dict(label=label,comparator='U',weaker_side=None if weak is None else targets[weak],target_rgb_mse={a:{b:parts[label,a][b]['pixel_weighted_mse'] for b in targets} for a in ['U','P','E','Q']},**outcome(checks)))
        for comparator in ['P','E']:
            q=[parts[label,'Q'][b]['coverage'] for b in targets];other=[parts[label,comparator][b]['coverage'] for b in targets]
            checks=[prior.comparison_check('min_target_side_coverage','drop_max',-.005,min(q) if all(v is not None for v in q) else None,min(other) if all(v is not None for v in other) else None)]
            checks+=protection(label,'Q',comparator,metrics,parts)
            target_checks.append(dict(label=label,comparator=comparator,**outcome(checks)))
    q_control=[c for c in comparisons if c['candidate']=='Q' and c['comparator']=='U'];eligible=all(c['numeric_pass'] for c in q_control+target_checks)
    save(NEW/'gate_results.json',dict(state='fixed_numeric_evaluated',protection_comparisons=comparisons,target_comparisons=target_checks,reference_denominator_checks=denominators,A_confirmation=dict(triggered=eligible,decision='required_pending_execution' if eligible else 'not_triggered',reason='All preregistered numeric conditions pass' if eligible else 'At least one fixed target or protection condition failed/unknown; no extra seeds or thresholds'),engineering_not_statistical_significance=True))
    save(NEW/'research/aggregation.json',dict(state='passed',mean_definition='Original macro mean per valid slot per identity; no reference-mask erasure of predictions',part_denominators=denominators,worst25='Reference-pixel weighted over25 consecutive slots; valid slots reported,all-unknown never zero',rows=dict(global_rows=len(global_rows),full_part_time_rows=len(fullrows),worst_windows=len(worst))))
    print('alpha48 gates',[(a['label'],a['numeric_pass'],a['failed'],a['unknown']) for a in alpha],flush=True)
    print('Q targets',[(a['label'],a['comparator'],a['numeric_pass'],a['failed']) for a in target_checks],flush=True);print('A confirmation',eligible,flush=True)
if __name__=='__main__':main()
