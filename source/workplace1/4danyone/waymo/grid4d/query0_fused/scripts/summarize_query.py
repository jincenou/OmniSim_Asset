"""Measured costs, per-person temporal curves, exposure and fixed cohort means."""
import collections,csv,json,os
from pathlib import Path
import numpy as np
from analyze_query import NEW,ROOT,PLAN,read,csvrows,write,descriptor,PARTS,FIELDS
from runtime import save
os.environ['MPLCONFIGDIR']=str(NEW/'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    globalrows=csvrows(NEW/'paired_metrics.csv');partrows=csvrows(NEW/'part_time_metrics.csv');means=[]
    for cohort,labels,methods in [('all_eight',PLAN['labels'],['HUST','F','alpha48','A']),('main_seven',[l for l in PLAN['labels'] if l!='N11'],['HUST','F','alpha48','A']),('reserve_N11',['N11'],['HUST','F','alpha48','A']),('query_three',PLAN['query_labels'],['F','U','P','E','Q','A'])]:
        for method in methods:
            rows=[r for r in globalrows if r['label'] in labels and r['method']==method];assert len(rows)==len(labels)
            means.append(dict(cohort=cohort,method=method,people=len(rows),**{f:float(np.mean([float(r[f]) for r in rows])) for f in FIELDS}))
    write(NEW/'cohort_metrics.csv',means);curves=[]
    for label in PLAN['query_labels']:
        for arm in ['U','P','E','Q']:
            for step in [500,1000,2000]:
                m=read(NEW/'results'/f'{label}_fused_{arm}_{32999+step}'/'metrics.json');curves.append(dict(label=label,arm=arm,extra_updates=step,actual_updates=32999+step,**m['mean']))
    write(NEW/'stage_metrics.csv',curves)
    behavior=[];costs=[];queue=read(NEW/'STATUS.json');shared=[]
    for label in PLAN['query_labels']:
        candidate=read(NEW/'data'/label/'candidates.json');initial=read(NEW/'demand_tables'/f'{label}_fused_initial.json');shared.append(dict(label=label,candidate_CPU_seconds=candidate['seconds'],initial_scoring_seconds=initial['seconds'],initial_render_calls=initial['render_calls'],shared_by_four=True))
        frames=candidate['frame_count'];views=candidate['train_views'];fig,ax=plt.subplots(4,4,figsize=(20,15),constrained_layout=True);fig.suptitle(f'{label}: training exposure / part-time selections / fixed heldout temporal cores',fontsize=16)
        targetparts=['left_leg','right_leg'] if label=='N06' else ['left_arm','right_arm']
        for col,arm in enumerate(['U','P','E','Q']):
            name=f'{label}_fused_{arm}';run=NEW/'runs'/name;trace=[json.loads(x) for x in (NEW/'sampling_traces'/f'{name}.jsonl').read_text().splitlines()];exp=read(run/'exposure.json');done=read(run/'complete.json');count=done['costs'];stages=queue['jobs'][name]['stages'];wall=lambda s:s['ended_at']-s['started_at']
            trainstage=next(s for s in stages if s['name'].endswith('_train'))
            costs.append(dict(run=name,label=label,arm=arm,loop_seconds=count['training_loop_seconds'],snapshot_seconds=count['snapshot_seconds'],diagnostic_refresh_seconds=count['diagnostic_seconds'],entry_seconds=done['end_to_end_seconds'],loading_and_other_seconds=done['end_to_end_seconds']-sum(count[k] for k in ['training_loop_seconds','snapshot_seconds','diagnostic_seconds']),train_process_seconds=wall(trainstage),replay_process_seconds=sum(wall(s) for s in stages if s['name'].endswith('_validate')),metrics_and_parts_process_seconds=sum(wall(s) for s in stages if s['name'].endswith(('_metrics','_parts'))),training_renders=count['training_render_calls'],refresh_renders=count['diagnostic_render_calls'],backward_calls=count['backward_calls'],optimizer_updates=count['optimizer_updates'],peak_allocated_bytes=done['peak_allocated_bytes'],peak_reserved_bytes=done['peak_reserved_bytes'],gpu_uuid=done['device']['uuid'],shared_load=True))
            exposure=np.zeros((21,frames),int);cells=np.zeros((6,8),int)
            for r in exp['observations']:exposure[views.index(r['view']),r['frame']]=r['count']
            for r in trace:
                if r['selected_part'] is not None:cells[r['selected_part'],r['selected_bin']]+=1
            image=ax[0,col].imshow(exposure,aspect='auto',vmin=0,vmax=max(6,int(exposure.max())));ax[0,col].set_title(f'{arm}: exposure / 2000 updates');ax[0,col].set_xlabel('native frame');ax[0,col].set_ylabel('21 training views');fig.colorbar(image,ax=ax[0,col],shrink=.6)
            image=ax[1,col].imshow(cells,aspect='auto',vmin=0);ax[1,col].set_title(f'{arm}: selected part-time cells');ax[1,col].set_yticks(range(6),PARTS);ax[1,col].set_xlabel('time bin');fig.colorbar(image,ax=ax[1,col],shrink=.6)
            for row,part in enumerate(targetparts,2):
                for method,color in [('F','#999999'),('A','#222222'),(arm,'#c63737')]:
                    rs=[r for r in partrows if r['label']==label and r['method']==method and r['part']==part];ys=[]
                    for f in range(frames):
                        vals=[float(r['coverage']) for r in rs if int(r['frame'])==f and r['coverage']!=''];ys.append(np.mean(vals) if vals else np.nan)
                    ax[row,col].plot(range(frames),ys,label=method,color=color)
                ax[row,col].set_title(part+' soft coverage');ax[row,col].set_ylim(0,1.02);ax[row,col].set_xlabel('frame / 25 fps');ax[row,col].legend()
            refresh=read(NEW/'demand_tables'/f'{name}_1000.json');scores=[]
            for tag,table in [('initial',initial),('refresh1000',refresh)]:
                d=np.array([v for row in table['demand'] for v in row if v is not None]);probs=(d+.001)/(d+.001).sum();scores.append(dict(table=tag,valid_cells=len(d),demand_min=float(d.min()),demand_max=float(d.max()),probability_min=float(probs.min()),probability_max=float(probs.max()),normalized_entropy=float(-np.sum(probs*np.log(probs))/np.log(len(d))),part_probability=[float(np.nansum([0 if v is None else v+.001 for v in row])/(d+.001).sum()) for row in table['demand']]))
            behavior.append(dict(label=label,arm=arm,unique_observations=int((exposure>0).sum()),total_observations=21*frames,zero_exposure=int((exposure==0).sum()),max_exposure=int(exposure.max()),priority_part_counts=cells.sum(1).tolist(),opacity_active_initial=read(run/'restore_validation.json')['initial_opacity_active_gt0005'],opacity_active_final=trace[-1]['active_opacity_gt0005'],points=done['points'],demand=scores))
        fig.savefig(NEW/'research'/f'{label}_diagnostics.png',dpi=140);plt.close(fig)
    write(NEW/'costs.csv',costs);save(NEW/'research/sampling_behavior.json',dict(state='measured',rows=behavior,scope='Training-only diagnostics, no H ranking. Entropy is normalized by valid cell count; cell counts are only odd priority updates.'))
    save(NEW/'cost_summary.json',dict(state='measured',shared_preparation=shared,branch_training_process_seconds_range=[min(r['train_process_seconds'] for r in costs),max(r['train_process_seconds'] for r in costs)],train_queue_wall_seconds=queue['ended_at']-queue['started_at'],cached_parent_cost_not_recharged=True,diagnostics_for_U_P_not_required_in_production=True,frontend=dict(new_runs=0,new_seconds=0,prior_generation_cost='Not isolated by the available frontend timing metadata; NA rather than guessing. Frozen generated inputs reused.'),prefix=dict(new_runs=0,costs_reference=str(ROOT/'optimization_cross8_20261018/costs.csv'),prior_matched_pairs=str(ROOT/'optimization_cross8_20261018/cost_summary.json')),same_device_prior_probe=str(ROOT/'optimization_quality_v3_20261018/research/step_cost_probe.json'),scope='Shared-load wall time; entry includes imports/data/I/O; no pure GPU busy time or controlled speed ranking. Additional failed preflights and alpha-only evaluations remain in task inventories/logs.'))
    # Human-facing tables requested by user; eight-person average explicitly includes reserve N11.
    text=['# 当前指标：Query 三人对照与八人平均','2026-10-19（北京时间）。每个人等权平均；不把 N11 的 121 帧作为额外权重。参考是 4DAnyone 生成视频，H 是已使用过的开发视角，不是真实 Waymo 新视角真值。','']
    for scope,title in [('all_eight','八人平均（主七人 + 备用 N11）'),('query_three','Query 三人平均（N04 / N06 / N11）')]:
        text+=['## '+title,'','|方案|人数|全图 PSNR↑|前景 PSNR↑|全图 SSIM↑|裁剪 LPIPS↓|轮廓 IoU↑|轮廓外 alpha↓|','|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in means:
            if r['cohort']==scope:text.append(f"|{r['method']}|{r['people']}|{r['full_psnr']:.4f}|{r['foreground_psnr']:.4f}|{r['ssim']:.6f}|{r['crop_lpips']:.6f}|{r['silhouette_iou']:.6f}|{r['outside_alpha_mean']:.6f}|")
        text.append('')
    text+=['Q 相对同预算 U：前景 PSNR +0.011623 dB，SSIM -0.000132，LPIPS +0.000314（变差）。相对未续训 F：前景 PSNR +0.057104 dB，但其中普通续训 U 已贡献 +0.045481 dB，不能把全部变化归因于查询。','', '目标较差侧：N06 左腿 Q/U=0.953880/0.953229（+0.065117 个百分点）；N11 左臂 Q/U=0.801066/0.799493（+0.157307 个百分点）。两者均低于预设 +2 个百分点。覆盖来自保守核心区，不是整条腿/手臂完整率。','', 'Q 三人均通过相对 U 的数值保护线，但两人目标增益和超过 P/E 的要求均失败；Q 三人均未通过相对 A 的全部旧保护线。未触发 A+U/Q 四条条件实验。','', '[逐人结果](paired_metrics.csv) · [固定终态判据](gate_results.json) · [八人/主七人/N11/三人均值 CSV](cohort_metrics.csv) · [完整可视化](comparison/index.html)']
    (NEW/'METRICS.md').write_text('\n'.join(text)+'\n')
    print('Summary, costs and diagnostics complete',flush=True)
if __name__=='__main__':main()
