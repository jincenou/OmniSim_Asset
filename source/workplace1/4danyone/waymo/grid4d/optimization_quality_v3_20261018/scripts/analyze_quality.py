"""Aggregate fixed terminal metrics and predeclared protection gates; no tuning."""
import csv,json,math,hashlib
from pathlib import Path
import numpy as np
NEW=Path(__file__).resolve().parents[1];ROOT=NEW.parent;OLD=ROOT/'optimization_boundary_20261018';BASE=ROOT.parent/'random8/r20261015_113332'
PARTS=['head','torso','left_arm','right_arm','left_leg','right_leg'];FIELDS=['full_psnr','foreground_psnr','ssim','crop_lpips','silhouette_iou','outside_alpha_mean']
def read(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvrows(p):return list(csv.DictReader(Path(p).open()))
def writecsv(p,rows):
    with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def descriptors():
    result={}
    for label in ['N01','N04','N07','N11']:
        result[label,'HUST61']=(BASE/'samples'/label/'evaluations/4dgs_grid61/results.json',ROOT/'audit/metrics'/label/'4dgs_grid61.csv',None)
        if label in ['N01','N07']:
            for method,stage,it in [('Native','33k',33000),('alpha','alpha',48000),('A','A',53000),('Fused','fused',33000)]:
                key=f'{label}_{stage}_{it}';folder=NEW/'results'/key
                if (folder/'parts.csv').exists():result[label,method]=(folder/'metrics.json',folder/'parts.csv',folder/'regions.csv')
        else:
            result[label,'Native']=(ROOT/'results'/label/'33000/metrics.json',ROOT/'audit/metrics'/label/'grid4d_33k.csv',None)
            result[label,'alpha']=(ROOT/'optimization/results'/f'{label}_alpha/metrics.json',ROOT/'optimization/results'/f'{label}_alpha/grid4d_alpha.csv',None)
            for method in ['A','U','B']:result[label,method]=(OLD/'results/primary'/f'{label}_{method}/metrics.json',OLD/'results/primary'/f'{label}_{method}/parts.csv',OLD/'results/primary'/f'{label}_{method}/regions.csv')
    for label,method,it in [('N04','Bmatch',53000),('N11','C1',20),('N11','C2',20)]:
        folder=NEW/'results'/f'{label}_{method}_{it}'
        if (folder/'parts.csv').exists():result[label,method]=(folder/'metrics.json',folder/'parts.csv',folder/'regions.csv')
    return result
def part_summary(rows):
    out={}
    for part in PARTS:
        selected=[r for r in rows if r['part']==part and r['coverage']!=''];temporal=[r for r in selected if r['temporal_mse']!=''];area=sum(int(r['reference_core_pixels']) for r in selected)
        out[part]=dict(valid_slots=len(selected),pixels=area,coverage=float(np.mean([float(r['coverage']) for r in selected])) if selected else None,pixel_weighted_mse=sum(float(r['mse'])*int(r['reference_core_pixels']) for r in selected)/area if area else None,compensated_rmse=math.sqrt(float(np.mean([float(r['temporal_mse']) for r in temporal]))) if temporal else None,temporal_valid_fraction=float(np.mean([float(r['temporal_valid_fraction']) for r in temporal])) if temporal else None,temporal_slots=len(temporal))
    arms=[r for r in rows if r['part'] in ['left_arm','right_arm'] and r['mse']!=''];area=sum(int(r['reference_core_pixels']) for r in arms);out['combined_arms']=dict(pixels=area,mse=sum(float(r['mse'])*int(r['reference_core_pixels']) for r in arms)/area if area else None)
    return out
def comparison_check(name,rule,limit,a,b):
    value=None if a is None or b is None else a/b if rule=='ratio_max' and b else a-b if rule=='increase_max' else b-a if rule=='drop_max' else abs(a-b) if rule=='absolute_difference_max' else None
    return dict(metric=name,rule=rule,limit=limit,value=value,candidate=a,comparator=b,state='unknown' if value is None else 'pass' if value<=limit else 'fail')
def main():
    desc=descriptors();metrics={};parts={};rawparts={};rows=[];partrows=[];region_summary={}
    previous_regions=read(OLD/'research/region_summary.json')
    for d in previous_regions:region_summary[d['label'],d['method'],d['region']]=d
    for (label,method),(mp,pp,rp) in desc.items():
        d=read(mp);metrics[label,method]=d['mean'];pr=csvrows(pp);rawparts[label,method]=pr;parts[label,method]=part_summary(pr)
        updates=52999 if method in ['A','U','B','Bmatch','C1','C2'] else 47999 if method=='alpha' else 32999 if method!='HUST61' else 33000
        rows.append(dict(label=label,cohort='reserve' if label=='N11' else 'main',method=method,base_reconstruction_updates=updates,count_basis='HUST nominal coarse+fine; actual optimizer count not asserted' if method=='HUST61' else 'actual Grid4D updates; color updates separate',color_updates=3360 if method=='C1' else 19 if method=='C2' else 0,**{k:d['mean'][k] for k in FIELDS},metrics_file=str(mp)))
        for part in PARTS:partrows.append(dict(label=label,method=method,part=part,**parts[label,method][part]))
        if rp:
            regions=csvrows(rp)
            for region in ['interior','inner','outer','far']:
                selected=[r for r in regions if r['region']==region];area=sum(int(r['area']) for r in selected)
                region_summary[label,method,region]={k:sum(float(r[k])*int(r['area']) for r in selected if r[k]!='')/area for k in ['positive_excess_alpha','rgb_mse','mean_pred_alpha','undercoverage_alpha']}
    writecsv(NEW/'paired_metrics.csv',rows);writecsv(NEW/'comparison/part_metrics_summary.csv',partrows)
    # Require identical per-frame anatomical and temporal denominator definitions.
    denominator_checks=[]
    for label in ['N01','N04','N07','N11']:
        anchor=rawparts[label,'HUST61'];signature=lambda rs:[(r['view'],r['frame'],r['part'],r['reference_core_pixels'],r['temporal_valid_pixels']) for r in rs]
        for l,method in rawparts:
            if l==label:assert signature(rawparts[l,method])==signature(anchor),(l,method)
        denominator_checks.append(dict(label=label,state='passed',rows=len(anchor)))
    gates=[]
    if ('N04','Bmatch') in metrics:
        for comparator in ['alpha','A','U']:
            checks=[];a=metrics['N04','Bmatch'];b=metrics['N04',comparator]
            for f,rule,limit in [('outside_alpha_mean','ratio_max',.9),('crop_lpips','increase_max',.002),('foreground_psnr','drop_max',.2),('silhouette_iou','drop_max',.003),('ssim','drop_max',.001)]:checks.append(comparison_check(f,rule,limit,a[f],b[f]))
            for part in PARTS:checks.append(comparison_check('coverage_'+part,'drop_max',.01,parts['N04','Bmatch'][part]['coverage'],parts['N04',comparator][part]['coverage']))
            checks.append(comparison_check('far_positive_excess_alpha','increase_max',1e-6,region_summary['N04','Bmatch','far']['positive_excess_alpha'],region_summary['N04',comparator,'far']['positive_excess_alpha']))
            gates.append(dict(label='N04',candidate='Bmatch',comparator=comparator,checks=checks,numeric_pass=all(c['state']=='pass' for c in checks),visual='pending_or_unknown'))
    colors=[]
    for candidate in ['C1','C2']:
        if ('N11',candidate) not in metrics:continue
        checks=[];a=metrics['N11',candidate];b=metrics['N11','A'];p=parts['N11',candidate];q=parts['N11','A']
        for f,rule,limit in [('crop_lpips','increase_max',-.002),('foreground_psnr','drop_max',.1),('ssim','drop_max',.001),('silhouette_iou','absolute_difference_max',1e-5),('outside_alpha_mean','absolute_difference_max',1e-5)]:checks.append(comparison_check(f,rule,limit,a[f],b[f]))
        checks.append(comparison_check('combined_arms_pixel_weighted_mse','ratio_max',.9,p['combined_arms']['mse'],q['combined_arms']['mse']))
        for part in ['left_arm','right_arm']:checks.append(comparison_check(part+'_mse','ratio_max',1.,p[part]['pixel_weighted_mse'],q[part]['pixel_weighted_mse']))
        for part in PARTS:
            checks.append(comparison_check(part+'_coverage','absolute_difference_max',1e-5,p[part]['coverage'],q[part]['coverage']))
            checks.append(comparison_check(part+'_compensated_rmse','ratio_max',1.05,p[part]['compensated_rmse'],q[part]['compensated_rmse']))
        colors.append(dict(label='N11',candidate=candidate,comparator='A',checks=checks,numeric_pass=all(c['state']=='pass' for c in checks),visual='pending_or_unknown'))
    if ('N11','C1') in metrics and ('N11','C2') in metrics:
        checks=[comparison_check('combined_arms_pixel_weighted_mse','ratio_max',.95,parts['N11','C2']['combined_arms']['mse'],parts['N11','C1']['combined_arms']['mse']),comparison_check('crop_lpips','increase_max',0.,metrics['N11','C2']['crop_lpips'],metrics['N11','C1']['crop_lpips'])]
        colors.append(dict(label='N11',candidate='C2',comparator='C1',checks=checks,numeric_pass=all(c['state']=='pass' for c in checks),interpretation='Different batch sizes and update counts; independent benefit requires these gates in addition to versusA'))
    fusion=[]
    for label in ['N01','N07']:
        if (label,'Fused') not in metrics:continue
        a,b=metrics[label,'Fused'],metrics[label,'A'];checks=[]
        for f,rule,limit in [('foreground_psnr','drop_max',.2),('ssim','drop_max',.001),('crop_lpips','increase_max',.002)]:checks.append(comparison_check(f,rule,limit,a[f],b[f]))
        for part in PARTS:
            checks.append(comparison_check('coverage_'+part,'drop_max',.01,parts[label,'Fused'][part]['coverage'],parts[label,'A'][part]['coverage']))
            checks.append(comparison_check('temporal_rmse_'+part,'ratio_max',1.05,parts[label,'Fused'][part]['compensated_rmse'],parts[label,'A'][part]['compensated_rmse']))
        fusion.append(dict(label=label,candidate='Fused32999',comparator='A52999',checks=checks,numeric_pass=all(c['state']=='pass' for c in checks),visual='pending_or_unknown'))
    save(NEW/'gate_results.json',dict(state='numeric_evaluated_visual_review_pending',boundary=gates,color=colors,fusion=fusion,denominator_checks=denominator_checks,engineering_not_statistical_significance=True,parameter_selection_used_H=False))
    save(NEW/'color_comparison.json',dict(state='numeric_evaluated',comparisons=colors,parts={m:parts['N11',m] for m in ['A','C1','C2'] if ('N11',m) in parts},metric_protocol='Same fixed full-reference body cores and original per-part motion compensated RMSE; pooled arms MSE weighted by reference pixels; missing reference regions stay NA',training_curves={a:read(NEW/'runs'/f'N11_{a}/mse_curve.json') for a in ['C1','C2']}))
    save(NEW/'research/region_summary.json',[dict(label=l,method=m,region=r,**d) for (l,m,r),d in region_summary.items() if m in ['Bmatch','C1','C2','Fused']])
    common=[]
    for method in ['HUST61','A']:
        labels=['N01','N04','N07'];common.append(dict(method=method,labels=labels,**{f:float(np.mean([metrics[label,method][f] for label in labels])) for f in FIELDS}))
    save(NEW/'comparison/common_cohort.json',dict(main=common,reserve_N11={m:metrics['N11',m] for m in ['HUST61','A','C1','C2'] if ('N11',m) in metrics},no_seven_person_Grid4D_mean=True))
    print('Numeric comparisons',[(r['candidate'],r.get('label'),r['numeric_pass']) for r in gates+colors+fusion],flush=True)
if __name__=='__main__':main()
