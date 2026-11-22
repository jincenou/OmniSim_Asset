"""Read-only reuse of prior experiments; build aligned six-column comparisons.

No training, model rendering, or metrics inferred from compressed display videos.
Run with the eff4d environment. All new outputs stay inside eff4d.
"""
import csv
import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from data import ROOT, OLD, save_json, sha
from video_tools import Video, uint8

OUT = ROOT / 'comparisons/hust_ftgs_eff4d_v1'
HUST = OLD / 'alternatives/4dgaussians'
VIEWS = [0, 8, 16]
FRAMES = list(range(0, 61, 5))
KEYS = ['foreground_psnr', 'crop_lpips', 'silhouette_iou', 'outside_alpha_mean']
NAMES = ['FTGS（sigma8）', 'HUST 4DGaussians（默认）', 'HUST 4DGaussians（时间网格61）', '复旦原生4DGS', '复旦原生4DGS＋置信度加权']
HEADERS = ['Generated reference', 'FTGS (sigma8)', 'HUST 4DGaussians (default)', 'HUST 4DGaussians (grid61)', 'Fudan native 4DGS', 'Fudan 4DGS + confidence']
SUBTITLES = ['Same input / not real novel-view GT', '21 views x 61 times / 15k steps',
             '21 x 61 / 3k coarse + 30k fine', '21 x 61 / 3k coarse + 30k fine',
             '12 x 31 / 12k steps, 48k samples', '12 x 31 / 12k steps, 48k samples']
FONT = ImageFont.load_default(size=17)
SMALL = ImageFont.load_default(size=13)
SOURCES = {}


def record(path, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    if expected is not None:
        assert digest == expected, f'Source hash changed: {path}'
    if str(path) in SOURCES:
        assert SOURCES[str(path)] == digest
    SOURCES[str(path)] = digest
    return digest


def read(path):
    record(path)
    return json.loads(Path(path).read_text())


def decode(path, size):
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24', input_params=['-threads', '1'])
    try:
        meta = next(reader)
        assert meta['size'] == size and abs(meta['fps'] - 25) < 1e-6, (path, meta)
        count = 0
        for raw in reader:
            count += 1
            yield np.frombuffer(raw, np.uint8).reshape(size[1], size[0], 3)
        assert count == 61, (path, count)
    finally:
        reader.close()


def compose(arrays, label, view, frame, detail_boxes=None):
    canvas = Image.new('RGB', (2112, 736), '#17202f')
    draw = ImageDraw.Draw(canvas)
    for col, (array, title, subtitle) in enumerate(zip(arrays, HEADERS, SUBTITLES, strict=True)):
        x = col * 352
        draw.text((x + 7, 7), title, font=FONT, fill='white')
        draw.text((x + 7, 32), subtitle, font=SMALL, fill='#cbd5e1')
        draw.text((x + 7, 52), 'White display / gray metrics', font=SMALL, fill='#94a3b8')
        if detail_boxes is None:
            assert array.shape == (640, 352, 3)
            canvas.paste(Image.fromarray(array), (x, 72))
        else:
            for row, box in enumerate(detail_boxes):
                im = ImageOps.contain(Image.fromarray(array).crop(box), (352, 290), Image.Resampling.NEAREST)
                y = 72 + row * 320
                draw.text((x + 7, y + 3), ['Upper body / accessories', 'Lower body / feet proxy'][row], font=SMALL, fill='white')
                canvas.paste(im, (x + (352 - im.width) // 2, y + 26 + (290 - im.height) // 2))
    draw.text((7, 715), f'{label} | camera {view:02d} | frame {frame:02d}/60 | t={frame/25:.2f}s | Same view/time; unequal training budgets. No enhancement.', font=SMALL, fill='white')
    return np.asarray(canvas)


def metric_row(label, name, metrics, source, **budget):
    return {'object': label, 'method': name, **{k: float(metrics[k]) for k in KEYS},
            'metric_source': str(source), **budget}


def table_md(rows):
    lines = ['|对象|方法|前景 PSNR ↑|裁剪 LPIPS ↓|轮廓 IoU ↑|轮廓外 Alpha ↓|',
             '|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['object']}|{r['method']}|{r['foreground_psnr']:.3f}|{r['crop_lpips']:.5f}|{r['silhouette_iou']:.4f}|{r['outside_alpha_mean']:.5f}|")
    return '\n'.join(lines)


def table_html(rows):
    return '<table><thead><tr><th>对象</th><th>方法</th><th>前景 PSNR ↑</th><th>裁剪 LPIPS ↓</th><th>轮廓 IoU ↑</th><th>轮廓外 Alpha ↓</th></tr></thead><tbody>' + ''.join(
        f"<tr><td>{r['object']}</td><td>{html.escape(r['method'])}</td><td>{r['foreground_psnr']:.3f}</td><td>{r['crop_lpips']:.5f}</td><td>{r['silhouette_iou']:.4f}</td><td>{r['outside_alpha_mean']:.5f}</td></tr>" for r in rows) + '</tbody></table>'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    records, rows, videos = [], [], []
    for label, oid, slug in [('R07', 'OBJ_0018', 'r07'), ('M02', 'OBJ_0019', 'm02')]:
        clip = ROOT / 'objects' / oid / 'clips/C001'
        manifest = read(clip / 'manifest.json')
        previous = read(clip / '08_evaluation/comparison.json')
        b0root = Path(previous['B0']['source'])
        b0path = b0root / 'quality_metrics/metrics.json'
        b0 = read(b0path)
        hroots = [HUST / 'runs' / f'{slug}_61_recipe30k_v1', HUST / 'runs' / f'{slug}_61_grid61_recipe30k_v1']
        hresults = [read(p / 'evaluation/results.json') for p in hroots]
        hm = read(HUST / 'data' / f'{slug}_short61_24/fdanyone.json')
        assert Path(hm['source_data']).resolve() == Path(manifest['data_root']).resolve()
        assert hm['heldout_views'] == VIEWS and hm['frame_count'] == 61 and hm['fps'] == 25
        assert hm['width'] == 352 and hm['height'] == 640 and hm['time_denominator'] == 60
        eroots = [clip / f'05_reconstruction/{v}_S1_12k_v1' for v in ['E0', 'E1']]
        emetrics = [read(p / 'evaluation/metrics.json') for p in eroots]
        pairs = {(v, f) for v in VIEWS for f in FRAMES}
        inputs = {(r['view'], r['frame']): r['sha256'] for r in b0['inputs']}
        for p, result in zip(hroots, hresults, strict=True):
            assert result['sampled_frames'] == FRAMES and result['frame_count'] == 61 and result['fps'] == 25
            selected = [r for r in result['samples'] if r['background'] == .5]
            assert len(selected) == 39 and {(r['view'], r['frame']) for r in selected} == pairs
            for r in selected:
                assert r['input_sha256'] == inputs[r['view'], r['frame']]
            for v in VIEWS:
                assert result['crops'][str(v)] == b0['crops'][str(v)]
        for result in emetrics:
            selected = [r for r in result['rows'] if r['frame'] in FRAMES]
            assert len(selected) == 39 and {(r['view'], r['frame']) for r in selected} == pairs
            for r in selected:
                assert r['crop'] == b0['crops'][str(r['view'])]
        # Validate the actual common reference pixels, not just matching filenames.
        for v in VIEWS:
            for f in range(61):
                path = Path(manifest['data_root']) / 'images' / f'cam{v:02d}' / f'{f:06d}.png'
                digest = record(path, manifest['source_hashes'][str(path.resolve())])
                if f in FRAMES:
                    assert digest == inputs[v, f]
        rows.append(metric_row(label, NAMES[0], b0['aggregates']['validation_generated']['0.5'], b0path,
                               train_views=21, train_times=61, optimizer_steps=15000, rendered_training_samples=15000, points=b0['gaussians']))
        for name, p, result in zip(NAMES[1:3], hroots, hresults, strict=True):
            rows.append(metric_row(label, name, result['mean']['0.5'], p / 'evaluation/results.json',
                                   train_views=21, train_times=61, coarse_steps=3000, fine_steps=30000,
                                   optimizer_steps=33000, points=result['gaussians']))
        for name, p, result in zip(NAMES[3:], eroots, emetrics, strict=True):
            a = result['aggregates']['legacy13times']
            mapped = dict(zip(KEYS, [a['gray_foreground_psnr'], a['gray_crop_alex_lpips'], a['silhouette_iou'], a['outside_alpha_mean']]))
            complete = read(p / 'complete.json')
            rows.append(metric_row(label, name, mapped, p / 'evaluation/metrics.json', train_views=12, train_times=31,
                                   optimizer_steps=complete['steps'], rendered_training_samples=complete['rendered_samples'], points=complete['points']))
        destination = OUT / label
        destination.mkdir()
        checks = []
        for view in VIEWS:
            bvideo = OLD / ('diagnostics/weak_velocity/before_after' if label == 'R07' else 'diagnostics/m02_duration/comparison') / f'view{view:02d}.mp4'
            record(bvideo, previous['source_hashes'][str(bvideo)])
            hvideos = [p / 'evaluation' / f'view{view:02d}.mp4' for p in hroots]
            for p, result in zip(hvideos, hresults, strict=True):
                record(p, result['videos'][p.name]['sha256'])
            readers = [decode(bvideo, (1408, 664))] + [decode(p, (704, 640)) for p in hvideos]
            writer = Video(destination / f'view{view:02d}.mp4', 2112, 736)
            detail = Video(destination / 'details_view08.mp4', 2112, 736) if view == 8 else None
            max_mae = np.zeros(3)
            max_corrected_mae = np.zeros(3)
            codec_background_offsets = []
            for frame in range(61):
                old, hd, hg = [next(reader) for reader in readers]
                path = Path(manifest['data_root']) / 'images' / f'cam{view:02d}' / f'{frame:06d}.png'
                rgba = np.asarray(Image.open(path), np.float32) / 255
                target = uint8(rgba[..., :3] * rgba[..., 3:] + 1 - rgba[..., 3:])
                for i, reference in enumerate([old[24:, :352], hd[:, :352], hg[:, :352]]):
                    difference = reference.astype(float) - target.astype(float)
                    # Archived OpenCV mp4v shifts white by roughly (-4,-2,-5).
                    # Correct that constant only for the alignment diagnostic;
                    # the displayed pixels themselves are never adjusted.
                    offset = np.median(difference[rgba[..., 3] < .01], axis=0)
                    mae = float(np.abs(difference).mean())
                    corrected_mae = float(np.abs(difference - offset).mean())
                    assert mae < 8 and corrected_mae < 3, (label, view, frame, i, 'reference/time alignment failed', mae, corrected_mae)
                    max_mae[i] = max(max_mae[i], mae)
                    max_corrected_mae[i] = max(max_corrected_mae[i], corrected_mae)
                    if frame == 0:
                        codec_background_offsets.append(offset.tolist())
                preds = []
                for p in eroots:
                    image_path = p / 'evaluation/renders' / f'cam{view:02d}' / f'{frame:06d}.png'
                    record(image_path)
                    preds.append(np.asarray(Image.open(image_path).convert('RGB')))
                arrays = [target, old[24:, 1056:1408], hd[:, 352:], hg[:, 352:], *preds]
                full = compose(arrays, label, view, frame)
                writer.add(full)
                if frame in [0, 10, 20, 30, 40, 50, 60]:
                    Image.fromarray(full).save(destination / f'view{view:02d}_frame{frame:03d}.png')
                if detail is not None:
                    ys, xs = np.where(rgba[..., 3] > .5)
                    x0, x1 = max(0, int(xs.min()) - 10), min(352, int(xs.max()) + 11)
                    y0, y1 = int(ys.min()), int(ys.max()) + 1
                    h = y1 - y0
                    boxes = [(x0, int(y0 + .15*h), x1, int(y0 + .75*h)),
                             (x0, int(y0 + .65*h), x1, min(640, y1 + 10))]
                    enlarged = compose(arrays, label, view, frame, boxes)
                    detail.add(enlarged)
                    if frame in [0, 30, 60]:
                        Image.fromarray(enlarged).save(destination / f'details_view08_frame{frame:03d}.png')
            for reader in readers:
                assert next(reader, None) is None, 'Extra source frames'
            videos.append(writer.close())
            if detail:
                videos.append(detail.close())
            checks.append({'view': view, 'frames': 61, 'reference_alignment_max_per_frame_MAE_uint8': max_mae.tolist(),
                           'max_codec_offset_corrected_MAE_uint8': max_corrected_mae.tolist(),
                           'first_frame_codec_background_offsets': codec_background_offsets,
                           'offset_only_used_for_alignment_check_not_display': True,
                           'source_order': ['FTGS', 'HUST default', 'HUST grid61'], 'target_check_thresholds_uint8': {'raw': 8, 'codec_offset_corrected': 3}})
            print(label, f'view{view:02d}', 'encoded and full decode verified', flush=True)
        records.append({'object': label, 'object_id': oid, 'common_metric_pairs': 39, 'crops': {str(v): b0['crops'][str(v)] for v in VIEWS},
                        'input_hashes_and_crops_match': True, 'video_alignment': checks})

    for row in rows:
        baseline = next(r for r in rows if r['object'] == row['object'] and r['method'] == NAMES[0])
        row['delta_vs_B0'] = {k: row[k] - baseline[k] for k in KEYS}
        row['lpips_reduction_vs_B0_percent'] = 100 * (1 - row['crop_lpips'] / baseline['crop_lpips'])
    for path, digest in SOURCES.items():
        assert sha(path) == digest, f'Source mutated during packaging: {path}'
    result = {'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              'state': 'metrics_aligned_videos_fully_decoded_visual_review_pending', 'rows': rows,
              'metric_protocol': {'views': VIEWS, 'frames': FRAMES, 'images_per_method': 39, 'resolution': [352, 640],
                                  'background': .5, 'mask_threshold': .5, 'crop': 'per-view union of 13 reference masks + 5% margin',
                                  'LPIPS': 'AlexNet; HUST/eff4d lpips0.1.4; FTGS TorchMetrics, archived single-pair equivalence error1.19e-7',
                                  'metric_provenance': 'Saved floating-point model evaluations; no metrics computed from display videos',
                                  'not_merged': ['SSIM: missing matched gray baseline/eff4d values', 'all61 averages: different evaluation subset']},
              'display': {'columns': HEADERS, 'background': 'white', 'fps': 25, 'frames': 61, 'sample_span_seconds': 2.4,
                          'encoding_duration_seconds': 2.44, 'full_panel_size': [352, 640], 'filtering_or_enhancement': False,
                          'sources': 'Reference RGBA; FTGS from archived four-panel MP4; HUST from original evaluation MP4s; Fudan native and confidence variants from evaluation PNGs. Prior compression remains in FTGS/HUST display.'},
              'checks': records, 'videos': videos, 'source_hashes': SOURCES, 'source_hashes_rechecked': True,
              'limits': ['Different training observations, loss, initialization, point counts and budgets; not representation-only ablation.',
                         'Generated heldout views already used for tuning, not independent real novel-view GT.',
                         'Grid61 is a later HUST walking candidate, not a globally approved default.'],
              'production_approved': False, 'new_training': False, 'new_gpu_rendering': False}
    save_json(OUT / 'comparison.json', result)
    with (OUT / 'metrics.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['object', 'method', *KEYS, 'lpips_reduction_vs_B0_percent', 'metric_source'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    note = '同口径：352×640、灰背景0.5、留出视角0/8/16、帧0/5/…/60，共39张图。指标来自原始浮点评测，视频为白背景展示。训练预算不一致，不能归因于单一后端表示。参考是生成观测，并非真实未见视角GT。'
    conclusion = 'HUST 4DGaussians时间网格61版在两对象的前景PSNR、裁剪LPIPS和轮廓外Alpha上均优于其余列；M02的IoU也最高。R07的IoU最高为复旦原生4DGS，说明感知细节与轮廓指标并不完全一致。HUST默认版同样在PSNR、LPIPS和轮廓外Alpha上优于两种复旦原生4DGS配方。生产视觉批准仍为0。'
    md = ['# FTGS / HUST 4DGaussians / eff4d 同屏对比', '', note, '', table_md(rows), '', conclusion, '',
          '视频从左到右：生成参考 / FTGS（sigma8） / HUST 4DGaussians（默认） / HUST 4DGaussians（时间网格61） / 复旦原生4DGS / 复旦原生4DGS＋置信度加权。视频中的Fudan即复旦。', '',
          'HUST两版均3k coarse+30k fine、21×61训练观测；FTGS为15k、21×61；复旦两版为12k、12×31、48k训练渲染。', '',
          '所有视频61帧、25fps，首末跨度2.4秒；保留原图大小，局部视频仅做相同裁剪及最近邻缩放。FTGS/HUST取自历史编码视频，保留压缩和颜色偏移；定量指标不从视频计算。', '',
          '[播放全部对照](index.html) · [CSV](metrics.csv) · [原始数值、差值与核验记录](comparison.json)', '']
    sections = []
    for label in ['R07', 'M02']:
        md += [f'## {label}', '']
        section = f'<section id="{label}"><h2>{label}</h2>'
        for name, title in [('view08', '视角08'), ('view00', '视角00'), ('view16', '视角16'), ('details_view08', '上身附件／下肢局部')]:
            poster = f'{label}/{name}_frame030.png'
            md.append(f'- [{title}视频]({label}/{name}.mp4) · [中间帧图片]({poster})')
            section += f'<h3>{title}</h3><video controls loop playsinline preload="metadata" poster="{poster}" src="{label}/{name}.mp4"></video><p><a href="{label}/{name}.mp4">打开视频</a> · <a href="{poster}">原尺寸图片</a></p>'
        section += '<details><summary>全部视角关键帧</summary>'
        for view in VIEWS:
            section += f'<p>视角{view:02d}：' + ' · '.join(f'<a href="{label}/view{view:02d}_frame{f:03d}.png">帧{f}</a>' for f in [0,10,20,30,40,50,60]) + '</p>'
        sections.append(section + '</details></section>')
        md.append('')
    (OUT / 'REPORT.md').write_text('\n'.join(md))
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FTGS · HUST · eff4d 对比</title><style>body{font:16px system-ui;background:#111827;color:#e5e7eb;max-width:2112px;margin:24px auto;padding:20px}a{color:#7dd3fc}p{line-height:1.7}video{width:100%;background:#17202f}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{border:1px solid #475569;padding:10px;text-align:right}td:nth-child(-n+2),th:nth-child(-n+2){text-align:left}.table{overflow:auto}section{margin-top:40px}nav{display:flex;gap:20px;flex-wrap:wrap}</style><h1>FTGS / HUST 4DGaussians / eff4d</h1>'
    page += '<nav><a href="#R07">R07 视频</a><a href="#M02">M02 视频</a><a href="REPORT.md">完整表格</a><a href="metrics.csv">下载 CSV</a><a href="comparison.json">数值与核验记录</a></nav>'
    page += f'<p>{note}</p><p><strong>{conclusion}</strong></p><div class="table">{table_html(rows)}</div>'
    page += '<p>六栏顺序：生成参考 → FTGS（sigma8） → HUST 4DGaussians（默认） → HUST 4DGaussians（时间网格61） → 复旦原生4DGS → 复旦原生4DGS＋置信度加权。视频中的Fudan即复旦。视频共61帧、25fps；建议全屏查看或打开原尺寸图片。</p><p>训练：FTGS为21×61、15k步；HUST两版为21×61、3k+30k步；复旦两版为12×31、12k步、48k训练渲染。FTGS/HUST显示来自已有MP4，复旦两版来自PNG，存在历史编码与颜色差异；所有指标都来自原始评测。</p>'
    page += ''.join(sections) + '</html>'
    (OUT / 'index.html').write_text(page)
    print(OUT / 'index.html', flush=True)


if __name__ == '__main__':
    main()
