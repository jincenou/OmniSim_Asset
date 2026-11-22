"""Fixed user gates, all arms/persons, original part aggregation retained."""
import json,csv,math,importlib.util,hashlib
from pathlib import Path
import numpy as np
NEW=Path(__file__).resolve().parents[1];ROOT=NEW.parent;BASE=ROOT.parent/'random8/r20261015_113332'
spec=importlib.util.spec_from_file_location('frozen_part_aggregation',ROOT/'optimization_quality_v3_20261018/scripts/analyze_quality.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
def read(p):return json.loads(Path(p).read_text())
def csvrows(p):return list(csv.DictReader(Path(p).open()))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write(p,rows):
    with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def check_ratio(name,a,b,limit):
    if b==0 and a is not None:
        state='pass' if a==0 and limit>=1 else 'unknown' if a==0 else 'fail'
        return dict(metric=name,rule='ratio_max',limit=limit,value=None,candidate=a,comparator=b,state=state,reason='zero denominator;zero/zero proves non-worsening but no relative improvement')
    return old.comparison_check(name,'ratio_max',limit,a,b)
def main():
    plan=read(NEW/'PLAN.json');labels=plan['labels'];arms=plan['arms'];metrics={};parts={};globalrows=[];partrows=[];expanded={};denoms=[];curves=[];costs=[];assets=[];behavior=[]
    for label in labels:
        refsignature=None
        for method in ['HUST','F','A']+arms:
            if method=='HUST':mp=BASE/'samples'/label/'evaluations/4dgs_grid61/results.json';pp=ROOT/'audit/metrics'/label/'4dgs_grid61.csv'
            elif method in ['F','A']:
                ref=plan['inputs'][label]['old_references']['fused' if method=='F' else 'A'];mp=Path(ref['metrics']);pp=Path(ref['parts'])
            else:mp=NEW/'results'/f'{label}_{method}_32999/metrics.json';pp=mp.parent/'parts.csv'
            m=read(mp)['mean'];rs=csvrows(pp);signature=[(r['view'],r['frame'],r['part'],r['reference_core_pixels'],r['temporal_valid_pixels']) for r in rs]
            if refsignature is None:refsignature=signature
            else:assert signature==refsignature,(label,method,'reference denominator mismatch')
            metrics[label,method]=m;parts[label,method]=old.part_summary(rs);globalrows.append(dict(label=label,method=method,**{k:m[k] for k in old.FIELDS}))
            for part in old.PARTS:partrows.append(dict(label=label,method=method,part=part,**parts[label,method][part]))
            if label in ['N06','N11']:
                key=f'{label}_{method}_32999' if method in arms else f'{label}_old_'+('fused' if method=='F' else method);expanded[label,method]=read(NEW/'results'/key/'expanded.json')['summary']
        denoms.append(dict(label=label,methods=['HUST','F','A']+arms,rows=len(refsignature),identical=True))
        for arm in arms:
            key=f'{label}_{arm}';run=NEW/'runs'/key;asset=NEW/'assets'/key;exp=read(asset/'export_validation.json');rep=read(asset/'replay_validation.json');assert exp['passed'] and rep['passed'];done=read(run/'training.complete.json');assert done['step']==32999
            for step in [5000,15000,18000,32999]:curves.append(dict(label=label,arm=arm,actual_updates=step,**{k:read(NEW/'results'/f'{key}_{step}/metrics.json')['mean'][k] for k in old.FIELDS}))
            snap=read(run/'snapshot_32999.json');counts=snap['counts'];assets.append(dict(key=key,path=str(asset),recovery=str(run/'recovery_32999.pt'),replay_passed=True,points=snap['points'],parameter_counts=exp['parameter_counts'],asset_bytes=sum(p.stat().st_size for p in asset.iterdir() if p.is_file())))
            costs.append(dict(key=key,label=label,arm=arm,logical_updates=32999,new_branch_updates=30000,main_entry_seconds=done['wall_seconds'],cumulative_loop_wall_with_shared_prefix=snap['training_wall_seconds'],load_seconds=snap['load_seconds'],target_renders=counts['target_renders'],alpha_renders=counts['alpha_renders'],support_time_queries=counts['support_times'],support_view_queries=counts['support_times']*4,support_evidence_tokens=counts['support_times']*128*32,training_image_IO_once=21*plan['inputs'][label]['frame_count'],terminal_bake_image_IO=4*plan['inputs'][label]['frame_count'] if arm in ['R2','R3'] else 0,peak_allocated_bytes=snap['peak_allocated'],peak_reserved_bytes=snap['peak_reserved'],gpu_uuid=snap['device']['uuid'],points=snap['points'],new_parameters=exp['parameter_counts']['new_query'],asset_bytes=assets[-1]['asset_bytes']))
            trace=[json.loads(x) for x in (run/'trace.jsonl').read_text().splitlines()];diags=[r['diagnostics'] for r in trace if r['diagnostics']]
            required=['query.weight','key.weight','value.weight','offset.weight','head.weight'] if arm=='R3' else ['head.weight'] if arm!='R0' else []
            behavior.append(dict(key=key,formal_gradient_max={k:max(r['query_grad_abs_sum'].get(k,0) for r in trace) for k in required},diagnostic_samples=len(diags),last_diagnostic=diags[-1] if diags else None,gaussian_group_update_counts=counts['gaussian_updates_by_group'],topology_events=sum(r['structure'] is not None for r in trace),attention_not_material_correspondence=True))
    write(NEW/'paired_metrics.csv',globalrows);write(NEW/'part_summary.csv',partrows);write(NEW/'stage_metrics.csv',curves);write(NEW/'costs.csv',costs);save(NEW/'ASSETS.json',assets);save(NEW/'research/formal_behavior.json',behavior)
    protect=[];effect=[];capacity=[];history=[]
    for label in labels:
        for comparator in ['R0','R1','R2','A','F','HUST']:
            a,b=metrics[label,'R3'],metrics[label,comparator];checks=[old.comparison_check(k,rule,limit,a[k],b[k]) for k,rule,limit in [('foreground_psnr','drop_max',.2),('ssim','drop_max',.001),('crop_lpips','increase_max',.002),('outside_alpha_mean','ratio_max',1.05)]]
            for part in old.PARTS:
                aa,bb=parts[label,'R3'][part],parts[label,comparator][part];checks.append(old.comparison_check('coverage_'+part,'drop_max',.01,aa['coverage'],bb['coverage']));checks.append(check_ratio('temporal_RMSE_'+part,aa['compensated_rmse'],bb['compensated_rmse'],1.05))
            r=dict(label=label,candidate='R3',comparator=comparator,checks=checks,numeric_pass=all(c['state']=='pass' for c in checks),visual='unknown_until_review')
            if comparator=='R0':protect.append(r)
            else:r['role']='descriptive comparison,not an additional user-specified admission gate';history.append(r)
        if label in ['N06','N11']:
            for comparator in ['R0','R2']:
                for scope in ['side0','side1','fixed_bad_window_left']:
                    a,b=expanded[label,'R3'][scope],expanded[label,comparator][scope];assert (a['known_rows'],a['roi_pixels'],a['reference_alpha_mass'])==(b['known_rows'],b['roi_pixels'],b['reference_alpha_mass'])
                    for route,elimit,dlimit in [('appearance',.95,1.),('completeness',1.,.90)]:
                        checks=[check_ratio('expanded_RGB_MSE',a['RGB_MSE'],b['RGB_MSE'],elimit),check_ratio('expanded_soft_undercoverage',a['soft_undercoverage'],b['soft_undercoverage'],dlimit)]
                        measured_pass=all(c['state']=='pass' for c in checks)
                        effect.append(dict(label=label,comparator=comparator,scope=scope,role='primary_left' if scope=='side0' else 'secondary_reference_scope',route=route,checks=checks,measured_scope_numeric_pass=measured_pass,numeric_candidate=measured_pass and a['unknown_rows']==0,full_scope_status='unknown' if a['unknown_rows'] else 'numeric_candidate' if measured_pass else 'fail',known_rows=a['known_rows'],unknown_rows=a['unknown_rows'],unmeasured_ranges_not_pass=True,visual='unknown_until_review'))
            a,b=expanded[label,'R3']['side0'],expanded[label,'R1']['side0'];capacity.append(dict(label=label,R3=a,R1=b,R1_matches_or_better_E_and_D=None if a['RGB_MSE'] is None or b['RGB_MSE'] is None else b['RGB_MSE']<=a['RGB_MSE'] and b['soft_undercoverage']<=a['soft_undercoverage'],percentage_gate=None))
    save(NEW/'gate_results.json',dict(state='numeric_only_visual_review_required',protection=protect,effectiveness=effect,capacity_control=capacity,descriptive_other_comparisons=history,denominator_checks=denoms,extra_R1_percentage_threshold=False,unknown_not_pass=True,user_corrected_MSE_not_RMSE=True))
    expandedrows=[dict(label=l,method=m,scope=s,**v) for (l,m),scopes in expanded.items() for s,v in scopes.items()];write(NEW/'expanded_summary.csv',expandedrows)
    means=[dict(method=m,people=3,**{k:float(np.mean([metrics[l,m][k] for l in labels])) for k in old.FIELDS}) for m in ['HUST','F','A']+arms];write(NEW/'cohort_metrics.csv',means)
    text=['# EVQ 固定终态指标','每个人等权。参考为生成 H 开发视角，不是真实 Waymo 未观测视角真值。新四组使用双视角监督，旧 A/F/HUST 是历史参照，算力预算不同。','','|方法|全图 PSNR↑|前景 PSNR↑|全图 SSIM↑|裁剪 LPIPS↓|轮廓外 alpha↓|','|---|---:|---:|---:|---:|---:|']
    for r in means:text.append(f"|{r['method']}|{r['full_psnr']:.4f}|{r['foreground_psnr']:.4f}|{r['ssim']:.6f}|{r['crop_lpips']:.6f}|{r['outside_alpha_mean']:.6f}|")
    text+=['','扩展 ROI 按人物、左右侧及固定坏窗口分别统计。外观路径要求 MSE 降5%且欠覆盖不恶化；完整性路径要求欠覆盖降10%且 MSE 不恶化。主目标是预先固定的左侧；不能用三人均值代替 N06/N11 的逐人结论。未知和未人工检查范围均未算通过。','', '[逐人指标](paired_metrics.csv) · [扩展区域](expanded_summary.csv) · [全部保护线](gate_results.json) · [完整对比视频](comparison/index.html)']
    (NEW/'METRICS.md').write_text('\n'.join(text)+'\n')
    from cost_report import main as cost_report
    cost_report()
    save(NEW/'numeric_analysis.complete.json',dict(passed=True,terminal_count=12,visual_review_complete=False));print('12 terminal numeric analysis complete',flush=True)
if __name__=='__main__':main()
