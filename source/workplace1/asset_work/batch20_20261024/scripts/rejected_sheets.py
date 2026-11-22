"""Build a portable gallery of existing rejected outputs, preserving audit decisions."""
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo
import csv
import hashlib
import json
import textwrap
import zipfile

from PIL import Image, ImageDraw, ImageFont, ImageOps

BASE = Path('/mnt/ojc/workplace1/asset_work/batch20_20261024/frontend_revision_01')
LIBRARY = Path('/mnt/ojc/workplace1/asset_library/manifests/asset_library_rigid.json')
OUT = BASE / 'pilot_results/rejected_sheets_239'
FONT_ROOT = Path('/mnt/ojc/miniconda3/envs/vllm_vlm/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def gate_failures(entry):
    audit = entry['quality']['audit']
    gates = [
        (not entry['quality']['rendering']['all_16_nonempty'], 'empty_or_invalid_render', '渲染未通过'),
        (not audit['keep'], 'audit_keep_false', '模型建议淘汰'),
        (not audit['category_match'], 'category_mismatch', '类别不一致'),
        (not audit['identity_consistent'], 'identity_mismatch', '实例外观不一致'),
        (audit['quality_score'] < 5, 'score_below_5', '质量分低于 5'),
    ]
    failed = [(key, label) for condition, key, label in gates if condition]
    assert failed, f'Reject record without a known quality gate: {entry["uid"]}'
    return failed


def main():
    library = json.loads(LIBRARY.read_text())['assets']
    entries = sorted((r for r in library if r['status'] == 'reject'),
                     key=lambda r: (r['dataset'], r['category'], r['provenance']['source_obj_id']))
    assert len(entries) == 239
    OUT.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(str(FONT_ROOT / 'DejaVuSans.ttf'), 19)
    small = ImageFont.truetype(str(FONT_ROOT / 'DejaVuSans.ttf'), 16)
    heading = ImageFont.truetype(str(FONT_ROOT / 'DejaVuSans-Bold.ttf'), 30)
    records, pages = [], []
    for start in range(0, len(entries), 8):
        selected = entries[start:start+8]
        canvas = Image.new('RGB', (2400, 1870), '#111a26')
        draw = ImageDraw.Draw(canvas)
        page = len(pages) + 1
        draw.text((26, 18), f'REJECTED OUTPUTS | {start+1:03d}-{start+len(selected):03d} / 239 | Page {page:02d}',
                  fill='white', font=heading)
        draw.text((26, 61), 'Source foreground + four actual Gaussian views | Existing model decisions, not a new manual verdict',
                  fill='#b8c7d9', font=font)
        draw.text((26, 88), 'Rejection can be due to identity/category mismatch even if the quality score is high. Full audit text is in the gallery and INDEX.csv.',
                  fill='#b8c7d9', font=small)
        filename = f'rejected_{page:02d}.jpg'
        page_records = []
        for slot, entry in enumerate(selected):
            x = 18 + (slot % 2) * 1190
            y = 130 + (slot // 2) * 425
            draw.rounded_rectangle((x, y, x+1164, y+407), radius=12, fill='#202d3d')
            number = start + slot + 1
            oid = entry['provenance']['source_obj_id']
            failed = gate_failures(entry)
            audit = entry['quality']['audit']
            real_count = len(entry['provenance']['real_images'])
            draw.text((x+12, y+10), f'#{number:03d} {oid} | {entry["category"]} | {entry["route"]} real={real_count}',
                      fill='white', font=font)
            draw.text((x+12, y+41), f'REJECT | model score {entry["quality_score"]}/10 | ' + ', '.join(k for k, _ in failed),
                      fill='#ffb2ac', font=small)
            source_path = entry['provenance']['source_image']
            rendering = entry['quality']['rendering']
            with Image.open(source_path) as source:
                images = [source.convert('RGB')]
            with Image.open(rendering['evidence']) as source:
                sheet = source.convert('RGB')
                size = rendering['resolution']
                assert sheet.size == (size*4, (size+24)*4)
                for view in [2, 6, 10, 14]:
                    sx, sy = view % 4 * size, view // 4 * (size+24) + 24
                    images.append(sheet.crop((sx, sy, sx+size, sy+size)))
            for k, (image, label) in enumerate(zip(images, ['SOURCE', '45 deg', '135 deg', '225 deg', '315 deg'])):
                px = x+12+k*229
                draw.text((px+3, y+71), label, fill='#b5c7dc', font=small)
                panel = Image.new('RGB', (219, 205), 'white' if k == 0 else '#808080')
                thumb = ImageOps.contain(image, panel.size)
                panel.paste(thumb, ((panel.width-thumb.width)//2, (panel.height-thumb.height)//2))
                canvas.paste(panel, (px, y+97))
            summary = ' / '.join(entry.get('defects', [])[:2])
            lines = textwrap.wrap(summary, width=120)
            if len(lines) > 4:
                lines = lines[:4]
                lines[-1] = lines[-1][:116] + ' ...'
            for line, text in enumerate(lines):
                draw.text((x+12, y+315+line*20), text, font=small, fill='#d1dbe7')
            assert digest(source_path) == entry['provenance']['source_image_sha256']
            record = dict(index=number, uid=entry['uid'], obj_id=oid, dataset=entry['dataset'],
                category=entry['category'], route=entry['route'], real_count=real_count,
                status=entry['status'], quality_score=entry['quality_score'],
                failed_gates=[k for k, _ in failed], failed_gates_zh=[label for _, label in failed],
                defects=entry.get('defects', []), audit=audit,
                all_16_nonempty=rendering['all_16_nonempty'],
                source_image=source_path, source_sha256=digest(source_path),
                render_sheet=rendering['evidence'], render_sha256=digest(rendering['evidence']),
                sheet=filename, sheet_slot=slot+1)
            records.append(record)
            page_records.append(record)
        draw.text((26, 1840), 'Views are framed independently. These are generated outputs rejected by screening; the Gaussian payloads are retained.',
                  fill='#9fb0c4', font=small)
        path = OUT / filename
        canvas.save(path, quality=94, subsampling=0)
        pages.append(dict(file=filename, count=len(selected), indices=[r['index'] for r in page_records], sha256=digest(path)))
        print(filename, len(selected), flush=True)

    assert len(records) == len({r['uid'] for r in records}) == 239
    assert {r['uid'] for r in records} == {e['uid'] for e in entries}
    status = dict(Counter(e['status'] for e in library))
    manifest = dict(created_at_beijing=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        assets=239, sheets=len(pages), library_manifest_sha256=digest(LIBRARY),
        all_status_counts=status, rejected_by_dataset=dict(Counter(r['dataset'] for r in records)),
        rejected_by_category=dict(Counter(r['category'] for r in records)),
        source='Unmodified source foreground and existing 3DGS render evidence; no new generation or quality reclassification.',
        views_deg=[45, 135, 225, 315], figures=pages, records=records)
    write_json(OUT / 'manifest.json', manifest)
    with (OUT / 'INDEX.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['index', 'obj_id', 'dataset', 'category', 'route', 'real_count', 'quality_score',
                  'status', 'failed_gates_zh', 'defects', 'sheet', 'sheet_slot']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {k: record[k] for k in fields}
            row['failed_gates_zh'] = '；'.join(row['failed_gates_zh'])
            row['defects'] = ' | '.join(row['defects'])
            writer.writerow(row)

    body = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>239 个淘汰资产 · 原图与生成结果</title><style>body{max-width:1800px;margin:26px auto;padding:0 22px;background:#111a26;color:#eff4fa;font:17px/1.65 system-ui}a{color:#9dd2ff}img{width:100%;height:auto}article{margin:38px 0}summary{cursor:pointer}li{margin:9px 0}.note{background:#233147;padding:16px;border-radius:9px}.nav{line-height:2.2}.nav a{margin-right:14px}details{padding:12px;border-bottom:1px solid #43506a}</style>
<h1>239 个淘汰资产 · 专用拼图</h1><div class="note"><b>nuScenes：92 个；Waymo：147 个，共 239 个。</b><br>
共 30 张拼图，每张最多 8 个；每个对象包含原始前景和 45°、135°、225°、315° 四个真实高斯渲染视角。
这是已有自动审核结果，不是重新人工判定。质量分高仍可能因实例外观不一致而被淘汰；完整判断字段和模型缺陷描述见每页下方。
原模型仍保留，本次只生成拼图，没有改变资产状态。各资产独立取景，图片尺寸不代表实际尺度。</div>
<p><a href="INDEX.csv">下载编号与淘汰原因清单</a> · <a href="manifest.json">完整审查记录与图片溯源</a></p><nav class="nav">'''
    for i, page in enumerate(pages, 1):
        body += f'<a href="#page{i:02d}">第 {i} 张</a>'
    body += '</nav>'
    for i, page in enumerate(pages, 1):
        selected = [records[k-1] for k in page['indices']]
        body += f'<article id="page{i:02d}"><h2>第 {i} 张 · #{selected[0]["index"]:03d}–#{selected[-1]["index"]:03d}</h2><a href="{page["file"]}"><img loading="lazy" src="{page["file"]}" alt="淘汰资产第 {i} 张拼图"></a>'
        for record in selected:
            body += f'<details><summary>#{record["index"]:03d} {escape(record["obj_id"])} · {escape(record["category"])} · 分数 {record["quality_score"]} · {escape("；".join(record["failed_gates_zh"]))}</summary>'
            body += '<p>以下为原模型的缺陷描述，可能存在误判：</p><ul>'
            body += ''.join(f'<li>{escape(d)}</li>' for d in record['defects'])
            body += '</ul></details>'
        body += '</article>'
    (OUT / 'START_HERE.html').write_text(body + '</html>')
    (OUT / 'README.txt').write_text('239 个自动淘汰资产，30 张拼图。打开 START_HERE.html 查看全部。\n每格为原始分割前景与四个真实 Gaussian 视角。完整原因在网页及 INDEX.csv。\n这是自动审核记录，可能存在误判，不代表人工确认每份资产不可用。本次未删除或修改原资产。\n')
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != 'SHA256SUMS.txt')
    hashes = {p.name: digest(p) for p in files}
    (OUT / 'SHA256SUMS.txt').write_text(''.join(f'{h}  {name}\n' for name, h in hashes.items()))
    archive, temporary = OUT.with_suffix('.zip'), OUT.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=5) as bundle:
        for path in sorted(OUT.iterdir()):
            if path.is_file():
                bundle.write(path, f'{OUT.name}/{path.name}')
    with zipfile.ZipFile(temporary) as bundle:
        assert bundle.testzip() is None
        for name, expected in hashes.items():
            assert hashlib.sha256(bundle.read(f'{OUT.name}/{name}')).hexdigest() == expected, name
    temporary.replace(archive)
    print(json.dumps(dict(gallery=str(OUT / 'START_HERE.html'), archive=str(archive),
                          size_mib=round(archive.stat().st_size/1024**2, 2), assets=len(records),
                          sheets=len(pages), validation='PASS'), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
