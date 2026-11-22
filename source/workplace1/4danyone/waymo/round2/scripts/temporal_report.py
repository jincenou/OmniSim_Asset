"""Compare frozen temporal controls; never infer visual approval from metrics."""
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
B = R/'benchmark'

def read(path):
    return json.loads(path.read_text())

def main():
    protocol = hashlib.sha256((B/'protocol.json').read_bytes()).hexdigest()
    manifest = read(B/'temporal_manifest.json')
    lines = ['# 行走时间建模对照', '',
        '同一生成输入、留出视角 0/8/16、121 帧、352×640、白背景。指标衡量生成观测的一致性，不是真实新视角精度。单 seed；视觉复核仍待完成。', '',
        '|轨迹|配方|点数|PSNR ↑|LPIPS ↓|前景 PSNR ↑|轮廓 IoU ↑|外部 alpha/前景像素 ↓|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    comparisons = []
    for label in ('walk063_rootlocal24', 'batch_0084_twentyfour'):
        baseline = None
        for recipe in ('capacity50k', 'dense_relocate45k', 'dense_duration8'):
            job_id = f'{label}_{recipe}'
            d = read(B/'results'/job_id/'metrics.json')
            if d['state'] != 'complete' or d['protocol_sha256'] != protocol or d['evaluated_images'] != 363:
                raise ValueError(f'Incomplete or incompatible metrics: {job_id}')
            if recipe == 'capacity50k':
                baseline = d
            else:
                expected = next(x for x in manifest['jobs'] if x['id'] == job_id)
                if d['job'] != expected or d['inputs'] != baseline['inputs']:
                    raise ValueError('Checkpoint manifest or reference inputs differ')
                if d['gaussians'] != baseline['gaussians'] or d['steps'] != baseline['steps']:
                    raise ValueError('Temporal comparison budget mismatch')
                comparisons.append({'label':label, 'recipe':recipe,
                    'delta_from_dense_baseline':{k:d['aggregates'][k]-v for k,v in baseline['aggregates'].items()},
                    'lpips_relative_reduction_pct':100*(baseline['aggregates']['lpips']-d['aggregates']['lpips'])/baseline['aggregates']['lpips'],
                    'visual_approval':False})
            a = d['aggregates']
            lines.append(f'|{label}|{recipe}|{d["gaussians"]:,}|{a["full_psnr"]:.3f}|{a["lpips"]:.4f}|{a["foreground_psnr"]:.3f}|{a["silhouette_iou"]:.4f}|{a["outside_alpha_per_foreground_pixel"]:.5f}|')
        video = R/'comparisons'/f'{label}_temporal'/'validation.json'
        v = read(video)
        if len(v['videos']) != 3 or any(x['frames'] != 121 for x in v['videos'].values()):
            raise ValueError('Incomplete comparison videos')
    lines += ['', '两项改动分别比较，未合并：重分配停止步数 4500→45000；时间高斯标准差上限 8 帧（0.32 秒），它不是硬截断的可见寿命。后者也改变初始化 duration。', '',
        '[行走同步对照](../comparisons/walk063_rootlocal24_temporal/view08.mp4) · [0084 同步对照](../comparisons/batch_0084_twentyfour_temporal/view08.mp4)', '',
        '不要仅凭指标提高宣布动作真实或视觉通过；仍需检查手脚、遮挡和漂浮点。']
    result = {'state':'metrics_complete_visual_review_pending','protocol_sha256':protocol,'comparisons':comparisons}
    (B/'temporal_results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    (B/'TEMPORAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__ == '__main__':
    main()
