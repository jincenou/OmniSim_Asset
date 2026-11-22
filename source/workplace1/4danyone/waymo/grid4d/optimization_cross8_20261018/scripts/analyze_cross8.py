"""Fixed terminal eight-identity comparisons, no checkpoint/identity selection."""
import csv,hashlib,importlib.util,json,math
from pathlib import Path
NEW=Path(__file__).resolve().parents[1];ROOT=NEW.parent;V3=ROOT/'optimization_quality_v3_20261018';BASE=ROOT.parent/'random8/r20261015_113332'
from publish_cross8 import descriptors
spec=importlib.util.spec_from_file_location('previous_numeric_protocol',V3/'scripts/analyze_quality.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
FIELDS=prior.FIELDS;PARTS=prior.PARTS;PLAN=json.loads((NEW/'PLAN.json').read_text())
def read(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rows(p):return list(csv.DictReader(Path(p).open()))
def write(p,data):
    with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def main():
    result=[];parts=[];metrics={};partdata={};gates=[];denominators=[];assets=[]
    for label in PLAN['labels']:
        signature=None
        for method in ['HUST61','A','fused']:
            if method=='HUST61':mp=BASE/'samples'/label/'evaluations/4dgs_grid61/results.json';pp=ROOT/'audit/metrics'/label/'4dgs_grid61.csv'
            else:
                desc=descriptors(label,method);mp=desc['results']/'metrics.json';pp=desc['results']/'parts.csv'
                assets.append(dict(label=label,method=method,asset=str(desc['run']),results=str(desc['results']),reused=label+'_'+method in PLAN['reuse']))
            m=read(mp);metrics[label,method]=m['mean'];pr=rows(pp);partdata[label,method]=prior.part_summary(pr)
            sig=[(r['view'],r['frame'],r['part'],r['reference_core_pixels'],r['temporal_valid_pixels']) for r in pr]
            if signature is None:signature=sig
            else:assert signature==sig,(label,method,'denominator mismatch')
            result.append(dict(label=label,cohort='reserve' if label=='N11' else 'main',method=method,updates=52999 if method=='A' else 32999 if method=='fused' else 33000,count_basis='HUST nominal coarse+fine' if method=='HUST61' else 'actual optimizer updates',**{k:m['mean'][k] for k in FIELDS},source=str(mp)))
            for part in PARTS:parts.append(dict(label=label,method=method,part=part,**partdata[label,method][part]))
        denominators.append(dict(label=label,state='passed',rows=len(signature)))
        a,b=metrics[label,'fused'],metrics[label,'A'];checks=[]
        for f,rule,limit in [('foreground_psnr','drop_max',.2),('ssim','drop_max',.001),('crop_lpips','increase_max',.002)]:checks.append(prior.comparison_check(f,rule,limit,a[f],b[f]))
        for part in PARTS:
            p,q=partdata[label,'fused'][part],partdata[label,'A'][part]
            checks.append(prior.comparison_check('coverage_'+part,'drop_max',.01,p['coverage'],q['coverage']))
            checks.append(prior.comparison_check('temporal_rmse_'+part,'ratio_max',1.05,p['compensated_rmse'],q['compensated_rmse']))
        failed=[c['metric'] for c in checks if c['state']=='fail'];unknown=[c['metric'] for c in checks if c['state']=='unknown']
        gates.append(dict(label=label,cohort='reserve' if label=='N11' else 'main',checks=checks,failed=failed,unknown=unknown,numeric_state='fail' if failed else 'unknown_incomplete_coverage' if unknown else 'pass',visual='pending_or_unknown',decision='Not production accepted from numeric checks alone'))
    stage_rows=[]
    for label in PLAN['labels']:
        for method,count,draws in [('Native',32999,33000),('alpha',47999,48000),('A',52999,53000),('fused',32999,33000)]:
            if method in ['A','fused']:mp=descriptors(label,method)['results']/'metrics.json'
            elif label in PLAN['new_A_labels']:mp=NEW/'results'/f"{label}_{'33k_33000' if method=='Native' else 'alpha_48000'}"/'metrics.json'
            elif label in ['N01','N07']:mp=V3/'results'/f"{label}_{'33k_33000' if method=='Native' else 'alpha_48000'}"/'metrics.json'
            elif method=='Native':mp=ROOT/'results'/label/'33000/metrics.json'
            else:mp=ROOT/'optimization/results'/f'{label}_alpha/metrics.json'
            m=read(mp);frames=read(BASE/'four_dg/data'/label/'fdanyone.json')['frame_count']
            stage_rows.append(dict(label=label,method=method,actual_updates=count,frames=frames,rendered_RGB_observations=draws,mean_RGB_draws_per_train_camera_frame=draws/(21*frames),**{k:m['mean'][k] for k in FIELDS},source=str(mp)))
    write(NEW/'stage_metrics.csv',stage_rows)
    cohort=[]
    for name,labels in [('main7',PLAN['main_labels']),('reserve_N11',['N11'])]:
        for method in ['HUST61','A','fused']:
            cohort.append(dict(cohort=name,n=len(labels),labels=labels,method=method,**{k:sum(metrics[label,method][k] for label in labels)/len(labels) for k in FIELDS}))
    write(NEW/'paired_metrics.csv',result);write(NEW/'part_metrics.csv',parts)
    save(NEW/'cohort_metrics.json',dict(state='all_terminal_pairs_available',rows=cohort,averaging='Macro mean of per-identity metric means; seven main and one reserve separate; no omitted difficult identities.'))
    save(NEW/'gate_results.json',dict(state='fixed_terminal_numeric_evaluated',checks=gates,reference_denominator_checks=denominators,H_used_for_parameter_selection=False,unknown_policy='Preserve original reference ROI/flow unknowns. Unknown is never pass or zero error.',engineering_not_statistical_significance=True))
    save(NEW/'ASSETS.json',dict(state='terminal_asset_locations_recovery_audit_separate',assets=assets,contract='PLY+hash/deform+config+camera+matching native runtime; no arbitrary action driving or metric world coordinates.'))
    md=['# 八人物固定终态指标','', '主七人与备用N11分别统计。生成参考开发集H={0,8,16}，非真实Waymo新视角真值。↑越大越好，↓越小越好。','', '|队列|方法|全图PSNR↑|前景PSNR↑|SSIM↑|LPIPS↓|外部alpha↓|','|---|---|---:|---:|---:|---:|---:|']
    for r in cohort:md.append(f"|{r['cohort']}|{r['method']}|{r['full_psnr']:.4f}|{r['foreground_psnr']:.4f}|{r['ssim']:.5f}|{r['crop_lpips']:.5f}|{r['outside_alpha_mean']:.6f}|")
    md+=['','|对象|A前景PSNR|融合前景PSNR|变化dB|A LPIPS|融合LPIPS|数值保护线|','|---|---:|---:|---:|---:|---:|---|']
    for g in gates:
        l=g['label'];a,b=metrics[l,'A'],metrics[l,'fused'];md.append(f"|{l}|{a['foreground_psnr']:.4f}|{b['foreground_psnr']:.4f}|{b['foreground_psnr']-a['foreground_psnr']:+.4f}|{a['crop_lpips']:.5f}|{b['crop_lpips']:.5f}|{g['numeric_state']}|")
    md+=['','N02腿部ROI原协议为unknown，不补零、不改分母。数值通过与完整视觉验收分开；共享GPU耗时不能直接当算法速度。','', '[完整视频](comparison/index.html) · [逐人逐方法CSV](paired_metrics.csv) · [全套保护线](gate_results.json) · [部位/时间明细](part_metrics.csv)']
    (NEW/'METRICS.md').write_text('\n'.join(md)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    x=np.arange(len(PLAN['labels']))
    for j,(method,color) in enumerate([('HUST61','#8a8a8a'),('A','#2d936c'),('fused','#8064a2')]):
        for ax,metric,title in [(axes[0],'foreground_psnr','Foreground PSNR (dB), higher better'),(axes[1],'crop_lpips','Crop LPIPS, lower better')]:
            ax.bar(x+(j-1)*.24,[metrics[l,method][metric] for l in PLAN['labels']],width=.24,label=method,color=color)
            ax.set_xticks(x,labels=[l+('*' if l=='N11' else '') for l in PLAN['labels']]);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend();fig.suptitle('Fixed terminal results: generated-reference development evaluation | *N11 reserve')
    fig.savefig(NEW/'comparison/metrics.png',dpi=160);plt.close(fig)
    print('Eight terminal pairs analyzed',[(g['label'],g['numeric_state']) for g in gates],flush=True)
if __name__=='__main__':main()
