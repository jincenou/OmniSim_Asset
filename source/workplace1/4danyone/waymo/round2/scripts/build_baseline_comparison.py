"""Collect existing matched baseline metrics and media into a local report."""
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from urllib.parse import quote
import csv
import hashlib
import html
import json
import os

R = Path(__file__).resolve().parents[1]
A = R / 'alternatives/4dgaussians'
OUT = R / 'baseline_comparison'
KEYS = ['full_psnr', 'full_ssim', 'foreground_psnr', 'crop_lpips',
        'silhouette_iou', 'outside_alpha_mean']
TITLES = ['PSNR↑', 'SSIM↑', '前景PSNR↑', '裁剪LPIPS↓', 'IoU↑', 'mask外alpha↓']
NAMES = {'m02_short61_24': 'M02：行走／持续移动自车',
         'r07_short61_24': 'R07：提袋行走／自车近停',
         'batch_0021_rootlocal24': '0021：行走／转身',
         'walk063_rootlocal24': 'walk063：行走',
         'batch_0084_twentyfour': '0084：站立／少量迈步',
         'phone_full': 'phone：站立／手机'}
ORDER = list(NAMES)


def load(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def url(path, directory):
    return quote(os.path.relpath(path, directory), safe='/')


def number(value, precision=5):
    return '—' if value is None else f'{value:.{precision}f}'


def main():
    OUT.mkdir(exist_ok=True)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    sources, media, rows, cases = {}, {}, [], []

    def record(path, kind, expected=None):
        assert path.is_file() and path.stat().st_size > 0, path
        digest = sha(path)
        if expected is not None:
            assert digest == expected, path
        target = media if kind == 'media' else sources
        target[str(path.relative_to(R))] = {'sha256': digest, 'bytes': path.stat().st_size}

    current_path = A / 'reports/metrics_latest.json'
    current = load(current_path)
    record(current_path, 'source')
    frozen = {x['label']: x for x in current['rows']}
    temporal = load(A / 'reports/temporal_recipe_results.json')
    record(A / 'reports/temporal_recipe_results.json', 'source')
    native = {x['run']: x for x in temporal['rows'] + temporal['independent_validation']
              if x['control'] == 'grid61'}

    for label in ORDER:
        default = frozen[label]
        root = A / 'runs' / default['run']
        comparison = load(root / 'ftgs_comparison/validation.json')
        record(root / 'ftgs_comparison/validation.json', 'source')
        metrics_path = root / 'evaluation/results.json'
        metrics = load(metrics_path)
        record(metrics_path, 'source', comparison['new_results_sha256'])
        if label in ['phone_full', 'batch_0084_twentyfour']:
            old_path = R / 'quality' / ('baseline_' + label) / 'quality_metrics/metrics.json'
        else:
            old_path = R / 'reconstruction/outputs' / label / 'quality_metrics/metrics.json'
        original = load(old_path)
        record(old_path, 'source')
        assert original['checkpoint_sha256'] == comparison['ftgs_checkpoints'][original['checkpoint']]
        assert all(original['crops'][str(view)] == metrics['crops'][str(view)] for view in [0, 8, 16])
        original_inputs = {(s['view'], s['frame']): s['sha256'] for s in original['inputs']}
        assert all(original_inputs[s['view'], s['frame']] == s['input_sha256'] for s in metrics['samples'])
        original_gray = original['aggregates']['validation_generated']['0.5']
        variants = [('FTGS 原始5k', 'FreeTimeGsVanilla', 0, 5000, None, original_gray)]
        if label in ['m02_short61_24', 'r07_short61_24']:
            for title in ['FTGS15k sigma15', 'FTGS15k sigma8']:
                variants.append((title, 'FreeTimeGsVanilla', 0, 15000, None,
                                 comparison['gray_ftgs_variants'][title]))
            legend = '生成参考 → FTGS原5k → FTGS15k sigma15 → FTGS15k sigma8 → 4DGaussians默认25'
        elif label == 'batch_0021_rootlocal24':
            legend = '生成参考 → FTGS原5k → 4DGaussians默认25'
        else:
            variants.append(('FTGS 增密50k', 'FreeTimeGsVanilla', 0, 50000, None,
                             default['reference_metrics']))
            legend = '生成参考 → FTGS原5k → FTGS增密50k → 4DGaussians默认25'
        variants.append(('4DGaussians 默认25', '4DGaussians', 3000, 30000, 25, default))
        candidates = [x for x in native.values()
                      if load(A / 'runs' / x['run'] / 'asset.json')['label'] == label]
        assert len(candidates) <= 1
        control = candidates[0] if candidates else None
        groups = [(root / 'ftgs_comparison', '原baseline及完整后端配方', legend)]
        if control:
            control_root = A / 'runs' / control['run']
            proof_path = control_root / 'temporal_comparison/results.json'
            proof = load(proof_path)
            record(proof_path, 'source', control['evidence_sha256'])
            variants.append(('4DGaussians 网格61', '4DGaussians', 3000, 30000, 61,
                             {**control['after'], 'full_ssim': control['ssim']['after']['full_ssim']}))
            assert Path(proof['baseline']) == root
            groups.append((control_root / 'temporal_comparison', '同33k时间网格对照',
                           '生成参考 → 4DGaussians默认25 → 4DGaussians网格61'))
        case_rows = []
        for title, backend, coarse, fine, grid, values in variants:
            row = {'label': label, 'sample': NAMES[label], 'variant': title,
                   'backend': backend, 'coarse_steps': coarse, 'fine_steps': fine,
                   'total_steps': coarse + fine, 'temporal_grid': grid,
                   'frames': default['frames'], **{key: values.get(key) for key in KEYS}}
            rows.append(row)
            case_rows.append(row)
        selected = next(x for x in case_rows if x['variant'] == (
            '4DGaussians 网格61' if control and label != 'phone_full' else '4DGaussians 默认25'))
        media_groups = []
        for directory, title, order in groups:
            videos = []
            for view in [0, 8, 16]:
                path = directory / f'view{view:02d}.mp4'
                expected = comparison['videos'][path.name]['sha256'] if directory.name == 'ftgs_comparison' else None
                record(path, 'media', expected)
                videos.append(path)
            images = []
            for frame in [0, default['frames'] // 2, default['frames'] - 1]:
                path = directory / f'view08_frame{frame:03d}.jpg'
                record(path, 'media')
                images.append(path)
            media_groups.append({'title': title, 'order': order, 'videos': videos, 'images': images})
        cases.append({'label': label, 'name': NAMES[label], 'frames': default['frames'],
                      'rows': case_rows, 'original': case_rows[0], 'selected': selected,
                      'reference': default['reference_metrics'], 'reference_name': default['reference_backend'],
                      'control': control, 'media': media_groups})

    protocol = ('352×640，灰背景0.5，留出生成视角0/8/16各13个时刻；21视角训练。'
                '同一人物各版本的参考RGBA哈希、固定crop及checkpoint对应关系已核验。'
                '指标来自模型浮点渲染，不从压缩视频或图片反算。')
    limits = ('参考是生成validation，非真实Waymo新视角GT。FTGS5k、15k/50k和4DGaussians33k'
              '预算、初始化、容量及损失不同，属于完整配方比较，不是纯表示消融。'
              '网格25→61保持同33k、输入、seed和其余显式参数，但网格容量、离散正则和自适应点数会变化。'
              '旧FTGS在此灰背景/crop协议未计算SSIM，表中以“—”保留，不能拿旧白背景整图SSIM填入。'
              '画面对照以白背景显示，数值采用上述灰背景；两种显示不混算。')
    summary_header = '|样本|FTGS原5k LPIPS↓|已有FTGS参考↓|4DGS默认25↓|4DGS网格61↓|当前建议|比原5k下降|\n|---|---:|---:|---:|---:|---|---:|\n'
    summary_rows = []
    for c in cases:
        by_title = {x['variant']: x for x in c['rows']}
        reduction = 100 * (c['original']['crop_lpips'] - c['selected']['crop_lpips']) / c['original']['crop_lpips']
        grid = by_title.get('4DGaussians 网格61')
        c['lpips_reduction_from_original5k_percent'] = reduction
        summary_rows.append(f"|{c['name']}|{c['original']['crop_lpips']:.5f}|{c['reference']['crop_lpips']:.5f}（{c['reference_name']}）|{by_title['4DGaussians 默认25']['crop_lpips']:.5f}|{number(grid['crop_lpips']) if grid else '未实验'}|{c['selected']['variant']}|{reduction:.2f}%|")
    markdown = ['# 原baseline到当前配方：指标、视频和图片', '', f'整理时间：{stamp}；指标实验截至2026-10-14晚。', '',
                '[浏览器查看视频与图片](baseline_comparison/index.html) · [完整指标CSV](baseline_comparison/metrics.csv) · [机器可读记录](baseline_comparison/metrics.json)', '',
                '## 原5k到当前版本', '', protocol, '', limits, '', summary_header + '\n'.join(summary_rows), '',
                '这里同时列出原始5k和后续优化FTGS，避免把增密50k／sigma8的15k误叫原baseline。当前建议优先考虑裁剪LPIPS，未声称所有指标第一。', '',
                '## 同33k预算的时间网格变化', '',
                '|样本|PSNR 25→61↑|SSIM 25→61↑|前景PSNR 25→61↑|LPIPS 25→61↓|IoU 25→61↑|mask外alpha 25→61↓|LPIPS变化|',
                '|---|---:|---:|---:|---:|---:|---:|---|']
    for c in cases:
        x = c['control']
        if not x:
            continue
        b, f = x['before'], x['after']
        change = f"下降{x['lpips_reduction_percent']:.2f}%" if x['lpips_reduction_percent'] > 0 else f"变差{-x['lpips_reduction_percent']:.2f}%"
        markdown.append(f"|{c['name']}|{b['full_psnr']:.3f}→{f['full_psnr']:.3f}|{x['ssim']['before']['full_ssim']:.4f}→{x['ssim']['after']['full_ssim']:.4f}|{b['foreground_psnr']:.3f}→{f['foreground_psnr']:.3f}|{b['crop_lpips']:.5f}→{f['crop_lpips']:.5f}|{b['silhouette_iou']:.4f}→{f['silhouette_iou']:.4f}|{b['outside_alpha_mean']:.5f}→{f['outside_alpha_mean']:.5f}|{change}|")
    markdown += ['', 'phone是负结果，网格61的LPIPS更差，保留默认25。生产视觉批准仍0，活动脚、手、附件和背面仍有问题。', '',
                 '## 各版本完整指标', '', '|样本|版本|总步数|PSNR↑|SSIM↑|前景PSNR↑|LPIPS↓|IoU↑|mask外alpha↓|', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in rows:
        markdown.append('|'+ '|'.join([x['sample'], x['variant'], str(x['total_steps'])] +
            [number(x[k], 3 if 'psnr' in k else 4 if k in ['full_ssim', 'silhouette_iou'] else 5) for k in KEYS]) + '|')

    body = [f'<p>{html.escape(protocol)}</p><p class="scope">{html.escape(limits)}</p>',
            '<nav>' + ' '.join(f'<a href="#{c["label"]}">{html.escape(c["name"].split("：")[0])}</a>' for c in cases) + '</nav>',
            '<p><a href="../BASELINE_COMPARISON.md">Markdown报告</a> · <a href="metrics.csv">完整CSV</a> · <a href="metrics.json">JSON与来源</a> · <a href="../gallery/index.html">全部资产</a></p>']
    for c in cases:
        markdown += ['', f'## {c["name"]}', '', f'{c["frames"]}帧，25fps；下方为相同生成相机和相同模型时刻。']
        body.append(f'<section id="{c["label"]}"><h2>{html.escape(c["name"])}</h2><p>{c["frames"]}帧，25fps。当前建议：{html.escape(c["selected"]["variant"])}</p>')
        body.append('<div class="table"><table><thead><tr><th>版本</th><th>总步数</th>' + ''.join(f'<th>{x}</th>' for x in TITLES) + '</tr></thead><tbody>')
        for x in c['rows']:
            body.append('<tr><td>'+html.escape(x['variant'])+'</td><td>'+str(x['total_steps'])+'</td>'+
                        ''.join('<td>'+number(x[k], 3 if 'psnr' in k else 4 if k in ['full_ssim', 'silhouette_iou'] else 5)+'</td>' for k in KEYS)+'</tr>')
        body.append('</tbody></table></div>')
        for group in c['media']:
            video, middle = group['videos'][1], group['images'][1]
            markdown += ['', f'**{group["title"]}**：左到右为 {group["order"]}。', '',
                '视频：'+' · '.join(f'[视角{view:02d}]({url(path,R)})' for view,path in zip([0,8,16], group['videos'])), '',
                '图片：'+' · '.join(f'[{tag}]({url(path,R)})' for tag,path in zip(['首帧','中帧','末帧'],group['images'])), '',
                f'![{c["name"]} {group["title"]}中帧]({url(middle,R)})']
            body.append(f'<h3>{html.escape(group["title"])}</h3><p>左到右：{html.escape(group["order"])}</p>')
            body.append(f'<video controls playsinline preload="none" poster="{url(middle,OUT)}"><source src="{url(video,OUT)}" type="video/mp4"></video>')
            body.append('<p>其他视角：'+' · '.join(f'<a href="{url(path,OUT)}">{view:02d}</a>' for view,path in zip([0,8,16],group['videos']))+'</p><div class="images">')
            for tag, path in zip(['首帧','中帧','末帧'], group['images']):
                body.append(f'<figure><a href="{url(path,OUT)}"><img loading="lazy" src="{url(path,OUT)}" alt="{tag}"></a><figcaption>{tag}；点击原图放大</figcaption></figure>')
            body.append('</div>')
        body.append('</section>')
    markdown += ['', '## 其他基线与验收', '',
        '[原8轨迹白背景全时序baseline](benchmark/REPORT.md)是独立协议，不与本表灰背景裁剪指标混排。', '',
        '[完整性与来源哈希](baseline_comparison/validation.json)。本次只整理已有指标和原视频／图片，不启动训练、不改动或重新编码媒体。HTML引用服务器现有相对路径，不是含所有媒体的独立下载包。']
    (R / 'BASELINE_COMPARISON.md').write_text('\n'.join(markdown) + '\n')
    style = 'body{font:16px/1.65 system-ui,sans-serif;margin:24px auto;padding:0 20px;max-width:1500px;color:#183044;background:#f4f6f8}h1,h2,h3{line-height:1.3}a{color:#0758a6}nav{display:flex;gap:12px;flex-wrap:wrap}section{background:white;padding:20px;margin:24px 0;border-radius:12px}video{width:100%;max-height:720px;background:#eee}.table{overflow:auto}table{border-collapse:collapse;white-space:nowrap;font-size:14px}td,th{border:1px solid #ccd4dc;padding:8px}th{background:#eaf0f6}.images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;height:auto}figcaption{font-size:13px}.scope{background:#fff4d9;padding:12px}@media(max-width:700px){.images{grid-template-columns:1fr}}'
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>4DAnyone baseline对比</title><style>'+style+'</style><body><h1>原baseline → 当前配方</h1><p>'+html.escape(stamp)+'；6条轨迹，原始5k及各优化阶段。视频点击播放，不自动下载。</p>'+''.join(body)+'<p>技术完成不代表生产外观通过。文件依赖服务器现有目录，不公开托管、不包含原始数据下载包。</p></body></html>'
    (OUT / 'index.html').write_text(page)
    serial_cases = []
    for case in cases:
        serial_cases.append({**{k:v for k,v in case.items() if k not in ['media','control']},
            'media': [{**g, 'videos':[str(p.relative_to(R)) for p in g['videos']],
                       'images':[str(p.relative_to(R)) for p in g['images']]} for g in case['media']]})
    result = {'updated_at':stamp, 'protocol':protocol, 'limitations':limits, 'cases':serial_cases,
              'rows':rows, 'source_files':sources, 'media_files':media,
              'missing_ssim': 'Not calculated for FTGS in this gray/crop protocol;not borrowed from white/fullframe benchmark.',
              'production_approved':0, 'media_modified':False, 'gpu_training_started':False}
    (OUT / 'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (OUT / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    validation = {'updated_at':stamp,'state':'six_matched_original_baselines_metrics_and_existing_media_verified',
                  'distinct_tracks':len(cases), 'metric_rows':len(rows), 'media_files_checked':len(media),
                  'source_files_checked':len(sources), 'matched_original_checkpoint_and_RGBA_crops':True,
                  'source_json_hashes_and_backend_comparison_video_hashes_checked':True,
                  'report':str(R/'BASELINE_COMPARISON.md'), 'html':str(OUT/'index.html'),
                  'decode_scope':'Original generation-stage full-decode receipts retained;this packaging rechecks hashes,does not redo render/decode.',
                  'generator_sha256':sha(Path(__file__)), 'output_sha256':{
                      str(p.relative_to(R)):sha(p) for p in [R/'BASELINE_COMPARISON.md',OUT/'index.html',OUT/'metrics.json',OUT/'metrics.csv']}}
    (OUT / 'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:validation[k] for k in ['state','distinct_tracks','metric_rows','media_files_checked','report','html']},ensure_ascii=False))


if __name__ == '__main__':
    main()

