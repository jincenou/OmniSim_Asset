"""Fixed person-wise G0/G1/G2 gates; reuse historical denominators and aggregation."""
import csv,json,importlib.util
from pathlib import Path
N=Path(__file__).resolve().parents[1];ROOT=N.parent;OLD=ROOT/'early_visual_query_20261019'
s=importlib.util.spec_from_file_location('original_aggregation',ROOT/'optimization_quality_v3_20261018/scripts/analyze_quality.py');old=importlib.util.module_from_spec(s);s.loader.exec_module(old)
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write(p,rows):
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ratio(n,a,b,l):
 if a is not None and b==0:return dict(metric=n,candidate=a,comparator=b,rule='ratio_max',limit=l,value=None,state='pass' if a==0 and l>=1 else 'unknown' if a==0 else 'fail')
 return old.comparison_check(n,'ratio_max',l,a,b)
def main():
 plan=read(N/'PLAN.json');metrics={};parts={};expanded={};globalrows=[];partrows=[];erows=[];assets=[];costs=[];denoms=[];lastlegs=[]
 for label in ['N06','N11']:
  signature=None
  for method in ['G0','G1','G2','R1','R2','R3','A','HUST61']:
   if method.startswith('G'):r=N/'results'/f'{label}_{method}_32999';mp=r/'metrics.json';pp=r/'parts.csv';ep=r/'expanded.json'
   elif method in ['R1','R2','R3']:r=OLD/'results'/f'{label}_{method}_32999';mp=r/'metrics.json';pp=r/'parts.csv';ep=r/'expanded.json'
   elif method=='HUST61':mp=ROOT.parent/'random8/r20261015_113332/samples'/label/'evaluations/4dgs_grid61/results.json';pp=ROOT/'audit/metrics'/label/'4dgs_grid61.csv';ep=OLD/'results'/f'{label}_old_HUST/expanded.json'
   else:ref=plan['inputs'][label]['old_references']['A'];mp=Path(ref['metrics']);pp=Path(ref['parts']);ep=OLD/'results'/f'{label}_old_A/expanded.json'
   m=read(mp)['mean'];pr=list(csv.DictReader(pp.open()));sig=[(r['view'],r['frame'],r['part'],r['reference_core_pixels'],r['temporal_valid_pixels']) for r in pr]
   if signature is None:signature=sig
   else:assert sig==signature,(label,method,'denominators')
   if label=='N11':lastlegs.extend(dict(r,comparison_method=method) for r in pr if int(r['frame'])==plan['inputs'][label]['frame_count']-1 and r['part'] in ['left_leg','right_leg'])
   metrics[label,method]=m;parts[label,method]=old.part_summary(pr);expanded[label,method]=read(ep)['summary']
   globalrows.append(dict(label=label,cohort='difficult_reserve' if label=='N11' else 'main_diagnostic',method=method,**{k:m[k] for k in old.FIELDS}))
   partrows.extend(dict(label=label,method=method,part=p,**parts[label,method][p]) for p in old.PARTS)
   erows.extend(dict(label=label,method=method,scope=k,**v) for k,v in expanded[label,method].items())
   if method.startswith('G'):
    key=f'{label}_{method}';run=N/'runs'/key;asset=N/'assets'/key;done=read(run/'training.complete.json');snap=read(run/'snapshot_32999.json');ex=read(asset/'export_validation.json');rep=read(asset/'replay_validation.json');assert ex['passed'] and rep['passed']
    assets.append(dict(key=key,path=str(asset),recovery=str(run/'recovery_32999.pt'),points=snap['points'],parameter_counts=ex['parameter_counts'],replay_passed=True,continuous_color='evq_dc_table_v1',asset_bytes=ex['asset_bytes']))
    costs.append(dict(key=key,additional_updates=done['counts']['additional_updates'],logical_step=done['step'],training_entry_seconds=done['seconds'],load_seconds=done['load_seconds'],loop_seconds=done['loop_seconds'],points=snap['points'],peak_allocated=snap['peak_allocated'],peak_reserved=snap['peak_reserved'],gpu_uuid=done['device']['uuid'],query_parameters=ex['parameter_counts']['new_query'],asset_bytes=ex['asset_bytes'],target_renders=done['counts']['target_renders'],alpha_renders=done['counts']['alpha_renders']))
  denoms.append(dict(label=label,rows=len(signature),all_equal=True))
 gates=[]
 for label in ['N06','N11']:
  for candidate,control in [('G1','G0'),('G2','G0'),('G2','G1')]:
   a,b=metrics[label,candidate],metrics[label,control];checks=[old.comparison_check(k,r,l,a[k],b[k]) for k,r,l in [('foreground_psnr','drop_max',.2),('ssim','drop_max',.001),('crop_lpips','increase_max',.002)]];checks.append(ratio('outside_alpha',a['outside_alpha_mean'],b['outside_alpha_mean'],1.05))
   for part in old.PARTS:
    p,q=parts[label,candidate][part],parts[label,control][part];checks.append(old.comparison_check('coverage_'+part,'drop_max',.01,p['coverage'],q['coverage']));checks.append(ratio('motion_RMSE_'+part,p['compensated_rmse'],q['compensated_rmse'],1.05))
   a,b=expanded[label,candidate],expanded[label,control]
   for scope in a:
    assert tuple(a[scope][k] for k in ['known_rows','roi_pixels','reference_alpha_mass'])==tuple(b[scope][k] for k in ['known_rows','roi_pixels','reference_alpha_mass'])
   aw,bw=a['fixed_bad_window_left'],b['fixed_bad_window_left'];checks.append(old.comparison_check('bad_window_deficit','increase_max',.01,aw['soft_undercoverage'],bw['soft_undercoverage']));checks.append(ratio('bad_window_RGB_MSE',aw['RGB_MSE'],bw['RGB_MSE'],1.05))
   if label=='N06':checks.append(ratio('torso_core_RGB_MSE',parts[label,candidate]['torso']['pixel_weighted_mse'],parts[label,control]['torso']['pixel_weighted_mse'],1.05))
   routes=[];a,b=a['side0'],b['side0']
   for name,e,d in [('appearance',.95,1.),('completeness',1.,.9)]:
    c=[ratio('expanded_RGB_MSE',a['RGB_MSE'],b['RGB_MSE'],e),ratio('expanded_deficit',a['soft_undercoverage'],b['soft_undercoverage'],d)];routes.append(dict(route=name,checks=c,measured_pass=all(x['state']=='pass' for x in c),unknown_rows=a['unknown_rows']))
   gates.append(dict(label=label,candidate=candidate,control=control,protection=checks,protection_measured_pass=all(x['state']=='pass' for x in checks),routes=routes,unknown_not_pass=True,whole_scope_numeric_candidate=all(x['state']=='pass' for x in checks) and any(r['measured_pass'] and r['unknown_rows']==0 for r in routes) and aw['unknown_rows']==0,visual='pending_or_unknown'))
 history=read(N/'queue_history.json');write(N/'stage_costs.csv',[dict(key=h['key'],stage=h['stage'],wall_seconds=h['seconds'],physical_gpu=h['gpu']['index'],exit_code=h['exit_code']) for h in history]);save(N/'cost_summary.json',dict(training_entry_seconds_sum=sum(c['training_entry_seconds'] for c in costs),formal_added_updates=sum(c['additional_updates'] for c in costs),shared_saved_warmup_cost_excluded=True,stage_costs='stage_costs.csv',shared_GPU_load_not_controlled=True,no_algorithmic_speedup_claim=True))
 write(N/'last_frame_leg_core.csv',lastlegs);write(N/'paired_metrics.csv',globalrows);write(N/'part_summary.csv',partrows);write(N/'expanded_summary.csv',erows);write(N/'costs.csv',costs);save(N/'ASSETS.json',assets);save(N/'gate_results.json',dict(personwise=gates,denominators=denoms,reference='generated H development views,not real hidden-view truth',old_R1_A_descriptive_only=True))
 text=['# Primitive 局部外观读取：固定终态指标','N06 与困难对象 N11 分列；不以合并均值替代逐人判定。参考为4DAnyone生成视频的H开发观测。旧R1/R3/A仅作历史参照。','','|对象|方法|全图 PSNR↑|前景 PSNR↑|SSIM↑|裁剪 LPIPS↓|轮廓外alpha↓|','|---|---|---:|---:|---:|---:|---:|']
 for r in globalrows:text.append(f"|{r['label']}|{r['method']}|{r['full_psnr']:.4f}|{r['foreground_psnr']:.4f}|{r['ssim']:.6f}|{r['crop_lpips']:.6f}|{r['outside_alpha_mean']:.6f}|")
 text+=['','G0：原anchor R3重新训练；G1：primitive局部固定采样和简单融合；G2：primitive局部可学习偏移和查询。全部保留原位置残差，局部颜色替换原DC，不叠加。','','[局部与坏窗口指标](expanded_summary.csv) · [全部逐人判据](gate_results.json) · [真实耗时](costs.csv) · [完整比较](comparison/index.html)','', '新资产的连续颜色先对离散C_delta插值；旧资产采用插值Z后过非线性颜色decoder。离散终态指标可比，不能声称两版连续时间定义相同。']
 (N/'METRICS.md').write_text('\n'.join(text)+'\n');save(N/'numeric_analysis.complete.json',dict(passed=True,terminals=6,visual_complete=False))
if __name__=='__main__':main()
