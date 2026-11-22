"""Audit actual sampling, then compare identical generated reference frames."""
import json
from pathlib import Path
from statistics import mean
import numpy as np
from after_batch import ROUND as R, read, write, now


def main():
    root=R/'diagnostics/time_budget'
    queue=read(R/'time_budget_queue.json')
    if queue['state']!='time_budget_diagnostics_complete_review_pending':
        raise RuntimeError('Time-support queue is not complete')
    records=[];audits=[]
    for job in read(R/'time_budget_plan.json')['jobs']:
        folder=root/f'{job["label"]}_{job["support"]}'
        experiment=read(folder/'experiment.json');done=read(folder/'run_complete.json')
        counts=np.asarray(done['sampling']['counts_by_camera_and_frame'])
        frames=experiment['actual_training_frames']
        if counts.shape!=(24,121) or counts.sum()!=15000:raise ValueError('Incomplete sampling')
        if counts[[0,8,16]].sum()!=0:raise ValueError('Heldout view leaked')
        if set(np.flatnonzero(counts.sum(0)))!=set(frames):raise ValueError('Actual temporal sampling mismatch')
        if not done['frozen_parameters_unchanged']:raise ValueError('Frozen dynamics changed')
        single=read(R/'diagnostics/single_time'/f'{job["label"]}_f060/experiment.json')
        differences={k for k,v in experiment['full_config'].items() if v!=single['full_config'][k]}
        if differences!={'result_dir'}:raise ValueError(f'Unexpected config changes: {differences}')
        if experiment['source_sha256']!=single['source_sha256']:raise ValueError('Different initial checkpoint')
        audits.append({'job':job['id'],'steps':int(counts.sum()),'sampled_frames':frames,
                       'heldout_samples':0,'configuration_diff':['result_dir']})
        for path in sorted(folder.glob('eval_f*/metrics.json')):
            metrics=read(path);frame=metrics['frame']
            fixed=read(R/'diagnostics/single_time'/f'{job["label"]}_f{frame:03d}/metrics.json')
            for split in ('training_diagnostic','validation_generated'):
                baseline=[x for x in metrics['samples'] if x['background']==1 and x['split']==split and x['model']=='dense50k']
                current=[x for x in metrics['samples'] if x['background']==1 and x['split']==split and x['model']=='time_support']
                one=[x for x in fixed['samples'] if x['background']==1 and x['split']==split and x['model']=='single_time']
                if any(len(x)!=3 for x in (baseline,current,one)):raise ValueError('Wrong evaluation scope')
                refs=lambda rows:{x['view']:x['reference_sha256'] for x in rows}
                if refs(baseline)!=refs(current) or refs(baseline)!=refs(one):raise ValueError('Reference images differ')
                row={'job':job['id'],'frame':frame,'split':split,
                     'metrics_path':str(path),'single_time_metrics':str(R/'diagnostics/single_time'/f'{job["label"]}_f{frame:03d}/metrics.json')}
                for tag,items in [('dense50k',baseline),('same_budget_support',current),('single_time',one)]:
                    row[tag]={k:mean(x[k] for x in items) for k in ('full_psnr','foreground_psnr','lpips','silhouette_iou','outside_alpha_per_foreground_pixel')}
                records.append(row)
    write(root/'results.json',{'state':'sampled_metrics_complete_visual_review_pending','updated_at':now(),
        'actual_sampling_audits':audits,'results':records,
        'scope':'All variants add15k steps from dense50k and freeze temporal parameters. Equal optimizer steps are not equal updates per frame. Selected-frame generated-reference metrics do not establish real-world geometry or full motion quality.'})
    lines=['# 等额15k步时间范围诊断','',
           '共同起点为dense50k，学习率计划、点数、相机划分、冻结参数均相同；只改变实际训练时间范围。单时刻、21时刻窗口与121时刻的每帧更新次数不同，这正是本实验检验的时间拟合负担。',
           '','指标取3个训练/留出生成视角、白背景，不能当作真实Waymo新视角指标。原始JSON另保留黑/灰背景、前景误差与外部alpha。尚待视觉复核。','',
           '|实验|帧|划分|原50k PSNR / LPIPS|同预算时间范围 PSNR / LPIPS|单时刻15k PSNR / LPIPS|',
           '|---|---:|---|---|---|---|']
    for row in records:
        values=[f'{row[k]["full_psnr"]:.3f} / {row[k]["lpips"]:.4f}' for k in ('dense50k','same_budget_support','single_time')]
        lines.append(f'|{row["job"]}|{row["frame"]}|{row["split"]}|'+ '|'.join(values)+'|')
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'state':'report_complete','comparisons':len(records)}))


if __name__=='__main__':main()
