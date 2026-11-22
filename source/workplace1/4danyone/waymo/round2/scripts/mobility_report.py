"""Aggregate the completed movement controls against the frozen benchmark."""
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
B = R/'benchmark'

def read(path):
    return json.loads(path.read_text())

def main():
    digest = hashlib.sha256((B/'protocol.json').read_bytes()).hexdigest()
    manifest = read(B/'mobility_manifest.json')
    pairs = []
    lines = ['# 行走与迈步样本：新增 50k 对照', '',
             '四项训练、四项全时序评测及六个对照视频均完成。沿用冻结评测协议：3 个留出生成视角 × 121 帧，352×640、白背景、整图指标。参考是生成视频，不能作为真实新视角重建精度。', '',
             '|轨迹|配方|实际点数|PSNR ↑|SSIM ↑|LPIPS ↓|前景 PSNR ↑|轮廓 IoU ↑|外部 alpha/前景像素 ↓|',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for label in ('walk063_rootlocal24', 'batch_0084_twentyfour'):
        values = {}
        for recipe in ('q0', 'budget_half_50k', 'capacity50k'):
            d = read(B/'results'/f'{label}_{recipe}'/'metrics.json')
            if d['state'] != 'complete' or d['protocol_sha256'] != digest or d['evaluated_images'] != 363:
                raise ValueError('Incomplete or incompatible evaluation')
            if recipe != 'q0':
                expected = next(j for j in manifest['jobs'] if j['id']==f'{label}_{recipe}')
                if d['job'] != expected:
                    raise ValueError('Frozen manifest mismatch')
            values[recipe] = d
            a = d['aggregates']
            lines.append(f'|{label}|{recipe}|{d["gaussians"]:,}|{a["full_psnr"]:.3f}|{a["ssim"]:.4f}|{a["lpips"]:.4f}|{a["foreground_psnr"]:.3f}|{a["silhouette_iou"]:.4f}|{a["outside_alpha_per_foreground_pixel"]:.5f}|')
        base, long, dense = (values[x] for x in ('q0','budget_half_50k','capacity50k'))
        if not base['inputs']==long['inputs']==dense['inputs']:
            raise ValueError('Different reference inputs')
        delta = {key:dense['aggregates'][key]-long['aggregates'][key] for key in dense['aggregates']}
        pairs.append({'label':label,'q0':base['aggregates'],'q50':long['aggregates'],
                      'dense50':dense['aggregates'],'q50_to_dense50':delta,
                      'lpips_relative_reduction_pct':100*(long['aggregates']['lpips']-dense['aggregates']['lpips'])/long['aggregates']['lpips'],
                      'matched_reference_images':363,'visual_approval':False})
    lines += ['', '同为 50k，增加容量/初始化密度后：']
    for p in pairs:
        d=p['q50_to_dense50']
        lines.append(f'- {p["label"]}：PSNR {d["full_psnr"]:+.3f} dB，SSIM {d["ssim"]:+.4f}，LPIPS 相对下降 {p["lpips_relative_reduction_pct"]:.2f}%。')
    lines += ['', '增加密度还改变初始表面采样和近邻尺度，不是纯点数消融。5k 与 50k 采用各自训练计划，也不是严格续训。', '',
              '本次抽看 view08：行走样本第 60/120 帧中，高密度版本轮廓较 5k 集中，但手、脚、脸仍明显比生成目标模糊，脚附近还有拖影；0084 第 60 帧中外套纹理和背包轮廓较清楚，但背包旁及裤腿外仍有条纹漂浮点。抽样不等于全视频视觉通过。', '',
              '[行走对照视频](../comparisons/walk063_rootlocal24_validation/view08.mp4) · [0084 对照视频](../comparisons/batch_0084_twentyfour_validation/view08.mp4) · [逐项数值](mobility_results.json) · [完整进度及后续安排](../REPORT.md)', '',
              '下一步优先诊断行走时段的监督一致性、动态高斯时间支持与速度，再做单变量实验；目前没有证据支持仅继续加到 100k 就能解决拖影。']
    result={'state':'complete','protocol_sha256':digest,'additional_jobs':4,'original_frozen_jobs':16,
            'total_jobs':20,'distinct_baseline_tracks':8,'tracks_with_three_recipes':6,'paired_results':pairs,
            'scope':'generated-heldout consistency only; single-seed descriptive comparisons'}
    (B/'mobility_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (B/'MOBILITY_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'state':'complete','paired_results':pairs},ensure_ascii=False))

if __name__=='__main__':
    main()
