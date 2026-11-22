"""Create a self-contained offline group-meeting package from verified artifacts."""
import csv
import hashlib
import html
import json
import shutil
import zipfile
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'comparisons/hust_ftgs_eff4d_v1'
DEST = ROOT / 'deliverables/group_meeting_20261015_v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(relative, text):
    path = DEST / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def csv_file(relative, headers, rows):
    path = DEST / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def finalize():
    """Check offline dependencies and copied media, then verify a ZIP round trip."""
    class References(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links, self.ids = [], set()

        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in ['href', 'src', 'poster']:
                    self.links.append(value)
                if key == 'id':
                    self.ids.add(value)

    checked_links = 0
    for page in DEST.rglob('*.html'):
        parser = References()
        parser.feed(page.read_text())
        for link in parser.links:
            parsed = urlsplit(link)
            assert not parsed.scheme and not parsed.netloc, ('External resource', page, link)
            if parsed.path:
                target = (page.parent / unquote(parsed.path)).resolve()
                assert target.is_relative_to(DEST.resolve()) and target.is_file(), (page, link)
            elif parsed.fragment:
                assert parsed.fragment in parser.ids, (page, link)
            checked_links += 1
    ET.parse(DEST / '02_tables/metrics.svg')
    provenance = json.loads((DEST / '05_records/provenance.json').read_text())
    for copy_record in provenance['source_copies']:
        assert digest(DEST / copy_record['file']) == copy_record['sha256']
    assert len(list((DEST / '03_videos').glob('*.mp4'))) == 8
    assert len(list((DEST / '04_images').glob('*.png'))) == 8
    with (DEST / '02_tables/metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        assert len(list(csv.reader(stream))) == 11
    check = {'checked_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'offline_local_links_verified': checked_links,
             'videos': 8, 'images': 8, 'metric_rows': 10, 'presentation_sections': 8,
             'media_copied_hash_identical': True, 'video_decoding': 'All8 source videos previously fully decoded; byte-identical copies verified.',
             'browser_runtime': 'No browser installed in build environment; static paths and embedded HTML logic inspected, not an automated browser playback test.',
             'svg_xml_valid': True, 'external_resource_dependencies': 0}
    write('05_records/package_checks.json', json.dumps(check, ensure_ascii=False, indent=2)+'\n')
    files = {}
    for file in sorted(DEST.rglob('*')):
        assert not file.is_symlink(), file
        if file.is_file() and file.name != 'MANIFEST.json':
            files[str(file.relative_to(DEST))] = {'sha256': digest(file), 'bytes': file.stat().st_size}
    write('MANIFEST.json', json.dumps({'files': files, 'scope': 'All package files except this manifest itself'}, ensure_ascii=False, indent=2)+'\n')
    archive = DEST.with_suffix('.zip')
    assert not archive.exists(), archive
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for file in sorted(DEST.rglob('*')):
            if file.is_file():
                output.write(file, Path(DEST.name) / file.relative_to(DEST))
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        with TemporaryDirectory(prefix='meeting-package-check-', dir='/tmp') as tmp:
            zipped.extractall(tmp)
            unpacked = Path(tmp) / DEST.name
            for name, entry in files.items():
                assert digest(unpacked / name) == entry['sha256'], name
            # Verify every relative HTML dependency again in the extracted tree.
            for page in unpacked.rglob('*.html'):
                parser = References()
                parser.feed(page.read_text())
                for link in parser.links:
                    path = unquote(urlsplit(link).path)
                    if path:
                        assert (page.parent / path).is_file(), (page, path)
    sha256 = digest(archive)
    archive.with_suffix('.zip.sha256').write_text(f'{sha256}  {archive.name}\n')
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size, 'files': len(files)+1,
                      'zip_crc_and_extracted_hashes_verified': True, 'sha256': sha256}, ensure_ascii=False), flush=True)


def main():
    data = json.loads((SOURCE / 'comparison.json').read_text())
    assert data['state'] == 'metrics_aligned_videos_fully_decoded_and_sampled_visual_review_complete'
    DEST.mkdir(parents=True, exist_ok=False)
    rows = data['rows']
    copied = []
    videos = {}
    images = {}
    expected_videos = {Path(v['path']).resolve(): v for v in data['videos']}

    def copy(source, relative, role):
        source = Path(source)
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = digest(source)
        if source.resolve() in expected_videos:
            assert expected == expected_videos[source.resolve()]['sha256']
        shutil.copyfile(source, target)
        assert digest(target) == expected == digest(source)
        copied.append({'source': str(source), 'file': relative, 'sha256': expected, 'role': role})
        return relative

    for label in ['R07', 'M02']:
        for view in [0, 8, 16]:
            name = f'{label}_models_view{view:02d}'
            videos[name] = copy(SOURCE / label / f'view{view:02d}.mp4', f'03_videos/{name}.mp4', 'six-model-column synchronized comparison')
        name = f'{label}_details'
        videos[name] = copy(SOURCE / label / 'details_view08.mp4', f'03_videos/{name}.mp4', 'matched upper/lower-region crops')
    selected = [('R07', 'view08', 30), ('R07', 'view00', 60), ('R07', 'view00', 0), ('R07', 'details_view08', 30),
                ('M02', 'view00', 30), ('M02', 'view08', 30), ('M02', 'view16', 30), ('M02', 'details_view08', 30)]
    for label, view, frame in selected:
        name = f'{label}_{view}_frame{frame:03d}'
        images[name] = copy(SOURCE / label / f'{view}_frame{frame:03d}.png', f'04_images/{name}.png', 'selected original-size comparison frame')

    headers = ['对象', '模型', '前景PSNR↑(dB)', '裁剪LPIPS↓', '轮廓IoU↑', '轮廓外平均Alpha↓']
    metric_rows = [[r['object'], r['method'], f"{r['foreground_psnr']:.3f}", f"{r['crop_lpips']:.5f}", f"{r['silhouette_iou']:.4f}", f"{r['outside_alpha_mean']:.5f}"] for r in rows]
    csv_file('02_tables/metrics.csv', headers, metric_rows)
    deltas = []
    for r in rows:
        if r['method'].startswith('FTGS'):
            continue
        delta = r['delta_vs_B0']
        base = next(x for x in rows if x['object'] == r['object'] and x['method'].startswith('FTGS'))
        deltas.append([r['object'], r['method'], f"{delta['foreground_psnr']:+.3f}", f"{r['lpips_reduction_vs_B0_percent']:+.2f}",
                       f"{100*delta['silhouette_iou']:+.2f}", f"{100*(1-r['outside_alpha_mean']/base['outside_alpha_mean']):+.2f}"])
    csv_file('02_tables/improvements_vs_FTGS.csv', ['对象', '模型', 'PSNR提升(dB)', 'LPIPS降低(%)', 'IoU提升(百分点)', '轮廓外Alpha降低(%)'], deltas)
    budget_headers = ['对象','模型','训练视角','训练时刻','不同观测数','优化步数','训练渲染次数','最终高斯点数']
    budgets = [[r['object'], r['method'], r['train_views'], r['train_times'], r['train_views']*r['train_times'],
                r['optimizer_steps'], r.get('rendered_training_samples', '此汇总未提供'), r['points']] for r in rows]
    csv_file('02_tables/training_budgets.csv', budget_headers, budgets)

    table_md = '|'+ '|'.join(headers)+'|\n|---|---|---:|---:|---:|---:|\n' + '\n'.join('|'+'|'.join(row)+'|' for row in metric_rows)
    protocol = '352×640；灰背景0.5；留出相机0、8、16；帧0、5、…、60；每种模型共39张图；固定前景裁剪。参考为生成观测，不是真实未见视角GT。'
    key_result = 'HUST 4DGaussians（时间网格61）在R07、M02的前景PSNR、裁剪LPIPS和轮廓外Alpha上均为本表最佳；M02的轮廓IoU也最高。R07的轮廓IoU以复旦原生4DGS略高。'
    write('02_tables/指标说明.md', '# 同口径指标表\n\n'+protocol+'\n\n'+table_md+'\n\n'+key_result+'''

前景PSNR和轮廓IoU越高越好；裁剪LPIPS和轮廓外Alpha越低越好。PSNR按dB报告；IoU差值用百分点。improvements_vs_FTGS.csv中正值统一表示改善，负值表示退步。

指标来自既有浮点模型评测，不从压缩视频计算。表中没有混入61时刻均值或不同背景SSIM。FTGS使用TorchMetrics AlexNet LPIPS，另两种后端使用lpips0.1.4 AlexNet；已有单对输入等价核查误差约1.19e-7，并非穷尽数值证明。

FTGS训练为21视角×61时刻、15k步；HUST两版为21×61、3k粗阶段+30k细阶段；复旦两版为12×31、12k步、48k训练渲染。初始化、损失和点数也不同。因此是完整配方效果对比，不是同预算的纯表示消融。HUST时间网格61是行走候选，不是全部动作统一默认。
''')

    talk = '''# 组会汇报讲稿：4DAnyone × Waymo 动态人物资产

建议时长：5–8分钟。材料整理日期：2026年10月15日。当前包聚焦R07、M02两条行走轨迹。

## 1. 任务与当前进度（约45秒）

我们希望从Waymo中的人物视频出发，生成多视角一致的人物运动，再重建为能够从不同视角、不同时间渲染的动态高斯资产。

此前Waymo阶段已经有18条不同轨迹完成技术链路，其中16条为121帧、2条为61帧；这是技术完成数量，不是18个独立自然人，也不代表视觉质量合格。HUST后端已有6条既有轨迹、14个完整实验版本。本次在R07和M02上进一步接入复旦原生4DGS，并测试置信度加权。生产视觉批准目前仍为0。

## 2. Pipeline与本次变化（约60秒）

共同前端为：实拍轨迹筛选和相机处理，RIFE时间插值与人物分割，GVHMR运动估计，分离水平根位移得到root-local动作，再用4DAnyone-Turbo生成24路同步视频，最后形成带相机和时间信息的RGBA。

本次复用已有生成结果，没有重新训练4DAnyone。比较的是三个重建后端：FTGS使用速度和时间持续性建模；HUST 4DGaussians使用规范高斯加时空变形；复旦使用联合空间—时间的原生4D高斯。复旦的置信度版本根据RIFE时间一致性对RGB损失加权，Alpha监督保持不变。本次没有启用额外扩散指导，也不是Efficient4D整篇论文复现。

## 3. 先看同屏效果（约90秒）

播放R07侧面完整动作，再播放M02正面和局部细节。每段都是61帧、25fps，首末跨度2.4秒，可以循环播放。

六栏从左到右是：生成参考、FTGS、HUST默认、HUST时间网格61、复旦原生4DGS、复旦原生4DGS加置信度。视频中的Fudan表示复旦。

抽样可见FTGS整体偏软；HUST更能保留衣服、包和腿脚细节，M02的网格61版本鞋部比默认版本更完整。但HUST仍然存在脚部破碎和局部轮廓伪影。复旦当前配方在衣服、包和下肢有明显细长条纹，加入置信度后没有稳定改善。

## 4. 定量结果（约60秒）

统一在三个留出生成视角、十三个时刻上评价，分辨率、背景和裁剪范围一致。HUST时间网格61版相对FTGS：R07的前景PSNR提升1.279dB，LPIPS降低44.73%；M02的PSNR提升1.537dB，LPIPS降低56.89%。

相对复旦原生4DGS，HUST时间网格61版的LPIPS在R07和M02上分别降低约47.00%和56.66%。但R07的轮廓IoU是复旦原生4DGS略高，因此不同指标并不完全一致。

目前不能说复旦适配版超过了旧工程的HUST路线。置信度模块确实进入了损失和梯度，但在这两个对象上没有显示一致收益。

## 5. 比较边界与当前问题（约60秒）

这些数值衡量对生成视频的重建一致性，不是真实Waymo未见视角的恢复精度。训练条件也不完全一致：FTGS是21×61观测、15k步；HUST是21×61、3k+30k步；复旦是12×31、12k步，但每步累积4张图，共48k训练渲染。模型点数、初始化、损失和颜色容量也不同，所以当前是配方对比，不能把提升唯一归因于表示。

展示R07末帧失败图片：HUST也有衣缘伪影，复旦条纹更明显。我们还没有完成生产质量验收、真实道路配准和可靠脚接地，也不支持任意换一个动作来驱动这个已重建人物。

## 6. 下一步（约45秒）

建议暂以HUST时间网格61作为行走场景的候选参考，同时保留站立场景默认配方。下一轮先固定预算和监督，集中检查腿脚和衣缘失败帧。对复旦方案，可只增加条件空间协方差轴比约束，检验细长高斯是否是主要原因；其他因素保持不变。

这些是后续建议，尚未在本包中执行新训练。当前优先目标是减少可见缺陷、补足真实视角验证，而不是扩大到100条资产生产。

## 备问答

问：训练的是4DAnyone吗？
答：不是。前端使用既有公开权重，训练是逐人物片段优化动态高斯资产。

问：为什么不只看PSNR？
答：轮廓、背景和模糊程度可能影响平均误差。应同时看LPIPS、轮廓IoU、轮廓外Alpha以及连续动作和局部失败。

问：61帧都是实拍吗？
答：不是。短片段源有25张实拍，输出的61个时间槽包含13个直接保留观测槽和48个插值槽；多视角参考来自生成结果。

问：这两个对象算独立测试吗？
答：不算。已用于调参；前端还可能跨视角交换信息。当前是留出生成观测的验证，而非独立真实新视角测试。

问：已完成Efficient4D全流程复现吗？
答：没有。当前是其公开复旦原生后端及时间置信度思路的本地适配，复用4DAnyone生成前端，没有运行额外扩散指导。

问：为什么视频背景略有颜色差异？
答：FTGS和HUST列来自历史MP4，复旦来自保存PNG，历史编码有轻微颜色偏移。没有做增强或校色；表格数值来自原始浮点评测。
'''
    write('01_text/汇报讲稿.md', talk)
    write('01_text/汇报讲稿.txt', talk.replace('## ', '').replace('# ', ''))
    summary = '''4DAnyone × Waymo 动态人物重建：当前情况摘要
整理日期：2026-10-15

任务：从真实人物轨迹，经运动估计和多视角生成，重建可按视角和时间重放的动态高斯资产。
前端：Waymo → RIFE/SAM2 → GVHMR/root-local → 4DAnyone-Turbo 24视角 → RGBA/相机/时间。
后端：FTGS；HUST 4DGaussians（默认、时间网格61）；复旦原生4DGS（等权、置信度加权）。

已完成：此前Waymo阶段18条不同轨迹技术链路；HUST在6条既有轨迹完成14个实验版本；本轮复旦适配在R07、M02完成基础及置信度两个版本、动态重载及评测。上述版本数不应相加当成人物数量。视觉生产批准仍为0。

核心结果：在本包两条行走轨迹、同一39张留出生成观测评测下，HUST时间网格61版的前景PSNR、裁剪LPIPS和轮廓外Alpha最佳。相对FTGS，R07的LPIPS降低44.73%，M02降低56.89%。R07的轮廓IoU以复旦基础配方略高；复旦置信度加权没有一致收益。

主要问题：FTGS偏模糊；HUST仍有脚部破碎及衣缘伪影；复旦当前配方存在明显细长拖影。平均指标改善不等于视觉质量已通过。

比较限制：参考是生成视角，不是真实未见视角GT；训练观测、优化预算、容量及损失不同，不能做纯后端因果归因；单seed且样本已用于调参。

下一步建议（未执行）：保持HUST行走候选参考，针对局部失败做受控实验；复旦可先单独验证协方差轴比正则；补充真实视角/配准/接地验证，暂不扩大生产队列。
'''
    write('01_text/一页摘要.txt', summary)

    def html_table(head, body):
        return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in body)+'</tbody></table></div>'

    table = html_table(headers, metric_rows)
    styles = '''*{box-sizing:border-box}body{margin:0;background:#101726;color:#ecf1f8;font:18px/1.65 system-ui,"Microsoft YaHei","PingFang SC",sans-serif}header,main,footer{max-width:1600px;margin:auto;padding:24px}a{color:#85cdfb}h1{font-size:38px;line-height:1.25}h2{font-size:29px;line-height:1.35}h3{font-size:21px}p{margin:12px 0}nav{display:flex;flex-wrap:wrap;gap:14px}section{padding:30px 0;border-top:1px solid #344155;margin:20px 0}.card{padding:20px;border-radius:12px;background:#1a2638}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}.tag{color:#6fe4c0;font-size:16px}.muted{color:#b1c1d5;font-size:15px}.warn{color:#ffd291}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:16px;font-variant-numeric:tabular-nums}td,th{padding:9px 12px;border:1px solid #3a4a60;text-align:right}td:nth-child(-n+2),th:nth-child(-n+2){text-align:left}th{background:#23334a}video,img{width:100%;height:auto;display:block;background:#17202f}figure{margin:20px 0}figcaption{font-size:15px;color:#b1c1d5;padding:8px 0}button{font:inherit;font-size:16px;padding:8px 16px;border:1px solid #64758e;border-radius:8px;background:#243b56;color:white;cursor:pointer}.controls{position:fixed;bottom:12px;right:18px;background:#101726ee;padding:8px;border-radius:10px;display:none;gap:12px;align-items:center}body.present header,body.present footer{display:none}body.present main{padding:20px 32px 90px;max-width:1920px}body.present section{display:none;border:0;margin:0;padding:0}body.present section.active{display:block}body.present .controls{display:flex}.pipeline{font-size:22px;padding:22px;border-left:5px solid #6fe4c0;background:#1a2638}@media print{body{background:white;color:black;font-size:12px}header nav,button,.controls,video{display:none!important}section{display:block!important;break-before:page}td,th{color:black;background:white}a{color:black}.card,.pipeline{background:#eee}.muted{color:#444}}'''
    def video(key, poster, caption):
        return f'<figure><video controls loop playsinline preload="metadata" src="{videos[key]}" poster="{images[poster]}"></video><figcaption>{caption} · <a href="{videos[key]}">单独打开MP4</a> · <a href="{images[poster]}">原尺寸图片</a></figcaption></figure>'
    legend = '从左到右：生成参考 → FTGS（sigma8） → HUST 4DGaussians（默认） → HUST 4DGaussians（时间网格61） → 复旦原生4DGS → 复旦原生4DGS＋置信度。视频中的 Fudan 即复旦。'
    slides = []
    slides.append('''<section id="overview"><p class="tag">01 / 任务与进度</p><h1>从人物视频到可重放的动态高斯资产</h1><p>4DAnyone × Waymo · 组会展示材料 · 2026-10-15</p><div class="grid"><div class="card"><h2>18 条轨迹</h2><p>此前Waymo阶段技术链路完成。版本数不等于人物数。</p></div><div class="card"><h2>3 类后端</h2><p>FTGS / HUST 4DGaussians / 复旦原生4DGS。</p></div><div class="card"><h2>2 个行走样本</h2><p>本包聚焦R07提袋行走、M02肩包行走。</p></div></div><p class="warn">视觉生产批准：0。技术链路完成与视觉验收分开报告。</p><p>当前结果：HUST时间网格61是这两个样本中更好的综合参考；复旦适配完成，但尚未带来一致质量优势。</p></section>''')
    slides.append('''<section id="pipeline"><p class="tag">02 / 流程与实验设计</p><h2>共同生成前端，比较不同重建后端</h2><div class="pipeline">Waymo轨迹 → 时间插值／人物分割 → GVHMR运动估计／root-local → 4DAnyone-Turbo生成24路视频 → RGBA、相机与时间 → 动态高斯重建</div><div class="grid"><div class="card"><h3>FTGS</h3><p>位置、线性速度、时间中心与持续时间。</p><p class="muted">21视角×61时刻，15k步。</p></div><div class="card"><h3>HUST 4DGaussians</h3><p>规范高斯＋时空变形；默认与时间网格61两个版本。</p><p class="muted">21×61，3k粗阶段＋30k细阶段。</p></div><div class="card"><h3>复旦原生4DGS</h3><p>联合空间与时间的高斯；另测RGB置信度加权。</p><p class="muted">12×31，12k步，48k训练渲染。</p></div></div><p>复用已有生成结果；训练是逐人物资产优化。没有重训前端，也没有运行额外扩散指导。</p><p class="warn">训练观测、预算、容量与损失不同：这是完整配方对比，不是纯表示消融。</p></section>''')
    slides.append('<section id="r07"><p class="tag">03 / R07：提袋行走</p><h2>衣服、手提袋和腿脚的动态对比</h2><p class="muted">'+legend+'</p>'+video('R07_models_view08','R07_view08_frame030','同一相机与时间，61帧／25fps；建议循环播放')+'<p>HUST保留更多衣服与包的细节；FTGS偏软；复旦两种配方存在细长条纹。</p></section>')
    slides.append('<section id="m02"><p class="tag">04 / M02：肩包行走</p><h2>正面连续动作与鞋部完整性</h2><p class="muted">'+legend+'</p>'+video('M02_models_view00','M02_view00_frame030','同一相机与时间，61帧／25fps')+'<p>HUST时间网格61版在抽样帧中鞋部更完整，但仍不能认定为生产质量通过。</p></section>')
    slides.append('<section id="metrics"><p class="tag">05 / 同口径指标</p><h2>HUST时间网格61在多数指标上领先</h2><p class="muted">'+protocol+'</p>'+table+'<p>相对FTGS：R07前景PSNR <strong>+1.279dB</strong>、LPIPS <strong>降低44.73%</strong>；M02分别为 <strong>+1.537dB</strong>、<strong>降低56.89%</strong>。</p><p class="warn">R07的轮廓IoU以复旦原生4DGS略高；不能只用一个指标评判视觉质量。</p><p><a href="02_tables/metrics.csv">指标CSV</a> · <a href="02_tables/improvements_vs_FTGS.csv">相对FTGS提升CSV</a> · <a href="02_tables/training_budgets.csv">训练预算CSV</a></p></section>')
    slides.append('<section id="details"><p class="tag">06 / 局部问题</p><h2>平均指标提升，仍伴随局部缺陷</h2>'+video('M02_details','M02_details_view08_frame030','相同裁剪与最近邻缩放；上排衣物／肩包，下排腿脚')+'<p>HUST仍有鞋部破碎、边缘颜色伪影；复旦有线状拖影；置信度加权未显示一致改善。</p></section>')
    slides.append('<section id="failure"><p class="tag">07 / 保留失败样例</p><h2>R07末帧：衣缘和身体周围仍有伪影</h2><figure><img src="'+images['R07_view00_frame060']+'" alt="R07末帧六栏对比，显示残留伪影"><figcaption>相机00，第60帧；不以滤点或图像增强隐藏缺陷。</figcaption></figure><p>HUST也有衣缘伪影，复旦条纹更明显；FTGS仍然模糊。</p><p class="warn">当前不扩大资产生产，不将这些结果描述为真实未见视角精确恢复。</p></section>')
    slides.append('''<section id="next"><p class="tag">08 / 结论与下一步</p><h2>保留当前参考配方，先解决局部质量</h2><div class="grid"><div class="card"><h3>已有结论</h3><p>HUST时间网格61是当前两个行走样本的较好参考；复旦适配链路可运行，置信度未带来稳定收益。</p></div><div class="card"><h3>下一轮建议，尚未执行</h3><p>固定监督和预算，针对脚部／衣缘做受控实验；复旦可先只测试协方差轴比约束。</p></div><div class="card"><h3>还需验证</h3><p>真实未见视角、源相机／世界坐标配准、脚接地，以及独立轨迹泛化。</p></div></div><p>不把已调参样本称为独立测试，不把技术完成当作生产验收。</p></section>''')
    extra = '<section id="backup"><h2>备用视频、图片与汇报文本</h2><p>演示模式默认跳过本页。所有链接均位于本文件夹，下载后无需服务器。</p><div class="grid">'
    for label in ['R07','M02']:
        extra += '<div class="card"><h3>'+label+'</h3>'
        for view in [0,8,16]:
            key=f'{label}_models_view{view:02d}';extra += f'<p><a href="{videos[key]}">相机{view:02d} 六栏完整对比</a></p>'
        extra += f'<p><a href="{videos[label+"_details"]}">局部细节对比</a></p></div>'
    extra += '</div><p><a href="01_text/汇报讲稿.txt">汇报讲稿TXT</a> · <a href="01_text/汇报讲稿.md">讲稿Markdown</a> · <a href="01_text/一页摘要.txt">一页摘要</a></p><details><summary>训练配置明细</summary>'+html_table(budget_headers,budgets)+'</details><details><summary>精选图片下载</summary>'+''.join(f'<p><a href="{path}">{name}</a></p>' for name,path in images.items())+'</details></section>'
    page = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>动态人物重建 · 组会展示</title><style>'+styles+'</style></head><body><header><nav><button onclick="start()">进入演示模式</button><button onclick="document.documentElement.requestFullscreen?.()">全屏</button><a href="#metrics">指标表</a><a href="#backup">备用素材</a><a href="01_text/汇报讲稿.txt">汇报讲稿</a><a href="02_tables/metrics.html">可打印指标表</a></nav><p class="muted">直接双击打开即可离线使用。演示模式用左右方向键翻页、Esc退出；视频点击播放并循环。浏览器缩放可调为80%–100%。</p></header><main>'+''.join(slides)+extra+'</main><footer><p>数据截至既有2026-10-14实验，材料于2026-10-15整理。本次只整理资料，没有新增训练。视频白背景展示，指标灰背景评价；FTGS/HUST列保留历史视频压缩及轻微颜色偏移。指标均来自原始浮点评测。</p></footer><div class="controls"><button onclick="move(-1)">上一页</button><span id="position"></span><button onclick="move(1)">下一页</button><button onclick="stop()">退出</button></div><script>const slides=[...document.querySelectorAll("main>section")].filter(x=>x.id!=="backup");let index=0;function show(){document.querySelectorAll("video").forEach(v=>v.pause());slides.forEach((s,i)=>s.classList.toggle("active",i===index));document.getElementById("position").textContent=(index+1)+" / "+slides.length;window.scrollTo(0,0)}function start(){document.body.classList.add("present");show()}function stop(){document.body.classList.remove("present");slides[index].scrollIntoView()}function move(n){index=Math.max(0,Math.min(slides.length-1,index+n));show()}document.addEventListener("keydown",e=>{if(!document.body.classList.contains("present"))return;if(e.key==="ArrowRight"||e.key==="PageDown"){e.preventDefault();move(1)}if(e.key==="ArrowLeft"||e.key==="PageUp"){e.preventDefault();move(-1)}if(e.key==="Escape")stop()});</script></body></html>'
    write('START_HERE.html', page)
    write('02_tables/metrics.html', '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>组会指标表</title><style>'+styles+'</style><main><h1>FTGS / HUST / 复旦原生4DGS 指标对比</h1><p>'+protocol+'</p>'+table+'<p>'+key_result+'</p><p>训练配方与预算不同；不作纯表示归因。生产批准：0。</p><button onclick="window.print()">打印／保存为PDF</button><p><a href="../START_HERE.html">返回展示页</a></p></main></html>')
    # Editable, resolution-independent table for direct insertion into slides.
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="850" viewBox="0 0 1600 850"><rect width="1600" height="850" fill="white"/><g font-family="Microsoft YaHei,PingFang SC,Arial,sans-serif" fill="#17202f">',
         '<text x="35" y="52" font-size="32">FTGS / HUST / 复旦原生4DGS：同口径指标</text>',
         '<text x="35" y="90" font-size="18">3个留出生成视角 × 13时刻；352×640；灰背景；训练预算不同。</text>']
    positions=[35,160,770,1000,1190,1380]
    for i,row in enumerate([headers]+metric_rows):
        y=140+i*54
        svg.append(f'<rect x="25" y="{y-31}" width="1550" height="54" fill="{("#e6eef5" if i==0 else "#f3f6f9" if i%2 else "white")}"/>')
        for x,value in zip(positions,row):svg.append(f'<text x="{x}" y="{y}" font-size="{18 if i==0 else 21}">{html.escape(value)}</text>')
    svg += ['<text x="35" y="775" font-size="20">PSNR、IoU越高越好；LPIPS、轮廓外Alpha越低越好。指标来自原始浮点评测。</text>',
            '<text x="35" y="812" font-size="20">生成观测验证，不是真实未见视角GT；生产视觉批准仍为0。</text></g></svg>']
    write('02_tables/metrics.svg',''.join(svg))
    readme='''组会展示包：4DAnyone × Waymo 动态人物重建
整理日期：2026-10-15

快速开始
1. 将ZIP完整解压，保留目录结构。不要只从压缩软件预览HTML。
2. 双击 START_HERE.html，用Chrome、Edge、Firefox或Safari打开。
3. 点击“进入演示模式”，左右方向键翻页；视频手动播放，可循环。Esc退出演示模式。
4. 如需放进自己的PPT，03_videos内MP4、04_images内PNG可直接插入。
5. 02_tables/metrics.svg是可缩放表格；metrics.html可直接打印为PDF；CSV含UTF-8 BOM，便于Excel打开。

目录
START_HERE.html：离线展示入口，8页建议展示内容，另有备用素材入口。
01_text：5–8分钟汇报讲稿（TXT/Markdown）、一页摘要。
02_tables：指标、相对FTGS提升、训练预算CSV；指标SVG、打印页与口径说明。
03_videos：8个六栏对比视频，两个对象各3个视角和1个局部细节视频。
04_images：8张精选原尺寸图片，包含效果与失败样例。
05_records：原始精度指标、来源及核验记录；供追溯，无需汇报时打开。
MANIFEST.json：包内文件SHA256，便于核验传输完整性。

建议播放顺序
R07_models_view08.mp4 → M02_models_view00.mp4 → 指标表 → M02_details.mp4 → R07_view00_frame060.png。
可补充R07_details.mp4以及两对象view16备用视角。

模型顺序
生成参考 / FTGS（sigma8） / HUST 4DGaussians（默认） / HUST 4DGaussians（时间网格61） / 复旦原生4DGS / 复旦原生4DGS＋置信度。
视频英文Fudan表示复旦；不用实验编号代替模型名称。

边界
所有文件已复制进本包，无符号链接、远程资源、服务器播放依赖或需要启动的服务。
视频首末跨度2.4秒，61帧/25fps编码时长2.44秒。均未增强、滤点或改变时间速度。
视频为白背景；表格为灰背景39张留出生成观测。FTGS/HUST来自已有MP4，存在编码差异；数值不来自视频。
三类后端训练预算和配方不同。技术完成不等于生产批准；当前批准数量仍为0。
素材用于本地组会展示，本次未上传或公开发布，不含原始Waymo实拍视频、训练权重或环境。
'''
    write('README.txt',readme)
    source_record = {'created_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'scope':'local meeting package; no new training; source artifacts copied without re-encoding',
                     'comparison_source_sha256':digest(SOURCE/'comparison.json'),'metrics_full_precision':rows,
                     'metric_protocol':data['metric_protocol'],'limits':data['limits'],'source_copies':copied,
                     'source_visual_review':data['visual_review'],'source_videos_full_decode':[{k:v[k] for k in ['path','frames','fps','sha256','full_decode']} for v in data['videos']],
                     'production_approved':False}
    write('05_records/provenance.json',json.dumps(source_record,ensure_ascii=False,indent=2)+'\n')
    print(DEST,flush=True)
    finalize()


if __name__=='__main__':main()
