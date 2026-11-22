"""Audit actual time sampling and summarize fixed-time diagnostic fits."""
import hashlib
import json
from pathlib import Path
import statistics

R=Path(__file__).resolve().parents[1]
D=R/'diagnostics/single_time'
def read(p):return json.loads(p.read_text())
def main():
    rows=[]
    lines=['# 固定时刻拟合诊断','',
        '从已有高密度50k模型出发，额外15k步只拟合一个时刻。时间中心、duration和速度冻结，关闭重分配，保留21训练视角和3留出视角。不是独立从零训练的静态3DGS，也不是完整动态资产。额外预算是混杂因素，不能单凭改善就宣称时间模型是唯一根因。','',
        '|样本 / 帧|视角组|基线 PSNR|单时刻 PSNR|基线 LPIPS|单时刻 LPIPS|LPIPS 相对下降|','|---|---|---:|---:|---:|---:|---:|']
    keys=['full_psnr','foreground_psnr','lpips','silhouette_iou','outside_alpha_per_foreground_pixel']
    for job in read(R/'single_time_plan.json')['jobs']:
        folder=D/f'{job["label"]}_f{job["frame"]:03d}'
        m=read(folder/'metrics.json');run=read(folder/'run_complete.json')
        if m['state']!='single_time_diagnostic_complete_visual_review_pending' or not run['frozen_parameters_unchanged']:
            raise ValueError('Incomplete diagnostic')
        if hashlib.sha256(Path(m['source_checkpoint']).read_bytes()).hexdigest()!=m['source_sha256']:
            raise ValueError('Source checkpoint changed')
        ledger=read(folder/'sampling_step14999.json')
        counts=ledger['counts_by_camera_and_frame'];frame=job['frame']
        if ledger['optimized_steps']!=15000 or sum(map(sum,counts))!=15000:
            raise ValueError('Unexpected actual training budget')
        if any(c for row in counts for index,c in enumerate(row) if index!=frame):
            raise ValueError('Other training times consumed')
        if any(sum(counts[v]) for v in (0,8,16)):
            raise ValueError('Heldout view training leakage')
        if any(sum(counts[v])==0 for v in ledger['train_camera_ids']):
            raise ValueError('Missing training view')
        for split in ('training_diagnostic','validation_generated'):
            values={}
            for model in ('dense50k','single_time'):
                s=[x for x in m['samples'] if x['split']==split and x['model']==model and x['background']==1.]
                if len(s)!=3 or any(x['frame']!=frame for x in s):raise ValueError('Metric coverage mismatch')
                values[model]={k:statistics.mean(x[k] for x in s) for k in keys}
            a,b=values['dense50k'],values['single_time'];reduction=100*(a['lpips']-b['lpips'])/a['lpips']
            rows.append({'id':job['id'],'frame':frame,'split':split,'white_background':values,
                'lpips_relative_reduction_pct':reduction,'actual_only_selected_time':True,'visual_approval':False})
            lines.append(f'|{job["id"]}|{split}|{a["full_psnr"]:.3f}|{b["full_psnr"]:.3f}|{a["lpips"]:.4f}|{b["lpips"]:.4f}|{reduction:.2f}%|')
    lines+=['','上述为白背景整图指标；黑/灰背景、前景与外部alpha原始数值保存在每项metrics.json。参考是生成图像，没有真实Waymo新视角真值。','',
        '解释：训练视角变清晰且留出视角也改善，说明当前参数仍有拟合空间；训练视角改善而留出不改善，应进一步查跨视角一致性与过拟合。改善不代表跨时间连贯性已解决。','',
        '[walk063帧30](walk063_rootlocal24_f030/view08.jpg) · [walk063帧60](walk063_rootlocal24_f060/view08.jpg) · [0084帧60](batch_0084_twentyfour_f060/view08.jpg)']
    result={'state':'sampling_audited_metrics_complete_visual_review_pending','rows':rows}
    (D/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (D/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
