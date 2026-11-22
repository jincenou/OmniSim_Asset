"""Live accounting and final previews for the two complete scene pilots."""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter
import csv
import html
import json
import os
from pathlib import Path
import sys
import time

B = Path(__file__).resolve().parents[1]
N = B / 'frontend_revision_01'
OUT = N / 'pilot_results'
LIBRARY = Path('/mnt/ojc/workplace1/asset_library')
P = Path('/mnt/ojc/pre_flux_pipeline')
sys.path.insert(0, str(P))
from asset_extraction.io_utils import read_json, write_json
from asset_extraction.generation import sha256

SCENES = {'nuscenes': 'scene-0001', 'waymo': 'waymo_val_000'}
NAMES = {'keep': '自动保留', 'review': '待复核', 'reject': '未通过', 'failed': '处理失败', 'pending': '进行中'}


def snapshot():
    OUT.mkdir(parents=True, exist_ok=True)
    chunks = sorted((N / 'chunks').glob('*'))
    generated, audited, publications = {}, {}, {}
    for chunk in chunks:
        state = chunk / 'generation/generation.json'
        if state.is_file():
            for row in read_json(state)['assets']:
                if row['obj_id'] in generated:
                    raise ValueError('Duplicate pilot query across chunks')
                generated[row['obj_id']] = row
        state = chunk / 'curation/curation.json'
        if state.is_file():
            audited.update(read_json(state)['assets'])
        state = chunk / 'published.json'
        if state.is_file():
            publications.update({r['source_obj_id']: r for r in read_json(state)})
        for attempt in sorted(chunk.glob('recovery_*')):
            state = attempt / 'generation/generation.json'
            if not state.is_file():
                continue
            for row in read_json(state)['assets']:
                uid = row['obj_id']
                if uid not in generated or generated[uid]['status'] == 'generated':
                    raise ValueError('Recovery must only replace a failed generation query')
                generated[uid] = row
                audited.pop(uid, None)
            state = attempt / 'curation/curation.json'
            if state.is_file():
                audited.update(read_json(state)['assets'])
            state = attempt / 'published.json'
            if state.is_file():
                publications.update({r['source_obj_id']: r for r in read_json(state)})
    batch = read_json(N / 'batch_status.json') if (N / 'batch_status.json').exists() else {}
    scenes, assets = [], []
    for dataset, scene in SCENES.items():
        run = N / f'runs/{dataset}_pilot'
        extraction = read_json(run / f'scenes/{scene}/status.json')
        rows = read_json(run / 'handoff/flux_input.json')
        routes = read_json(run / 'handoff/asset_routes.json')
        if sha256(run / 'handoff/flux_input.json') != routes['flux_input_sha256']:
            raise ValueError('Pilot cohort changed')
        by_id = {r['obj_id']: r for r in routes['assets']}
        selected = []
        for row in rows:
            uid = row['obj_id']
            gen, audit = generated.get(uid, {}), audited.get(uid, {})
            entry = audit.get('entry', {})
            state = audit.get('status', 'pending')
            rendering = entry.get('quality', {}).get('rendering', {})
            if not rendering and gen:
                # Audit errors must not erase an independently successful render.
                work = Path(gen.get('raw_path', '/absent')).parents[1] if gen.get('raw_path') else None
                if work:
                    root = work.parent / f'curation/asset_{gen["item_index"]:05d}'
                    render_file = root / 'canonical_render/render.json'
                    if not render_file.is_file():
                        render_file = root / 'axis_render/render.json'
                    if render_file.is_file():
                        rendering = read_json(render_file)
            record = dict(dataset=dataset, scene_id=scene, obj_id=uid, category=row['category'],
                route=by_id[uid]['route'], real_count=len(by_id[uid]['real_images']),
                native_foreground=by_id[uid]['source'].get('native_foreground'),
                source_image=row['image_path'], source_sha256=sha256(row['image_path']),
                flux_completed=gen.get('stages', {}).get('flux', {}).get('status') == 'succeeded',
                generated=gen.get('status') == 'generated', render_valid=rendering.get('all_16_nonempty', False),
                status=state, defects=entry.get('defects', []), error=audit.get('error'),
                audit_image=entry.get('quality', {}).get('audit_image'),
                canonical=entry.get('canonical'), published=publications.get(uid),
                quality_score=entry.get('quality_score'), review_reason=entry.get('reason'))
            if gen.get('status') == 'failed':
                record['status'] = 'failed'
                record['error'] = '; '.join(r.get('error', '') for r in gen['stages'].values() if r['status'] == 'failed') or record['error']
            selected.append(record)
        counts = Counter(r['status'] for r in selected)
        query_count = len(rows)
        removals = Counter(json.loads(line)['reason'] for line in
            (run / 'logs/rigid_quality_rejects.jsonl').read_text().splitlines())
        old = read_json(B / f'runs/{dataset}10/handoff/flux_input.json')
        summary = dict(dataset=dataset, scene_id=scene, images=extraction['num_images'],
            detections=extraction['num_raw_detections'], tracklets=extraction['num_tracklets'],
            sam_identity_budget=320, sam_observations=extraction['num_sam_candidates'],
            mask_pass=extraction['num_pre_vlm_kept'], source_quality_pass=extraction['num_kept'],
            source_quality_rejected=extraction['num_vlm_rejected'],
            before_revision_queries=sum(r['scene_id'] == scene for r in old), queries=query_count,
            post_source_removed=extraction['num_kept'] - query_count,
            post_source_removal_reasons=dict(removals),
            routes=dict(Counter(r['route'] for r in selected)),
            vehicle_real_counts=dict(Counter(r['real_count'] for r in selected if r['route'] == 'H')),
            flux_completed=sum(r['flux_completed'] for r in selected),
            generated=sum(r['generated'] for r in selected), render_valid=sum(r['render_valid'] for r in selected),
            audit_counts={name: counts[name] for name in NAMES},
            published=sum(r['published'] is not None for r in selected),
            kept_published=sum(r['status'] == 'keep' and r['published'] is not None for r in selected),
            generation_complete=(N / f'reports/{dataset}_generation_complete.json').is_file()
                and batch.get('stage') != 'recovery_running' and not counts['pending'])
        summary['aor_percent'] = 100 * summary['render_valid'] / query_count
        summary['bank_acceptance_percent'] = 100 * counts['keep'] / query_count
        summary['by_category']={category:dict(queries=len(items),generated=sum(r['generated'] for r in items),
            **{name:sum(r['status']==name for r in items) for name in NAMES})
            for category in sorted({r['category'] for r in selected})
            if (items:=[r for r in selected if r['category']==category])}
        gt_path = N / f'evaluation/matches/{scene}.json'
        if gt_path.is_file():
            gt = read_json(gt_path)
            summary['coverage'] = dict(denominator=gt['denominator_identities'], scope=gt['native_scope'],
                identity_scope=gt['identity_scope'], thresholds={})
            for threshold, match in gt['thresholds'].items():
                ids = {r['obj_id'] for r in selected if r['status'] == 'keep' and r['published']}
                linked = match['query']['matches']
                kept_gt = {r['gt_id'] for r in linked if r['obj_id'] in ids}
                summary['coverage']['thresholds'][threshold] = dict(
                    detector_identities=match['detector']['matched_identities'],
                    admitted_identities=match['query']['matched_identities'], kept_identities=len(kept_gt),
                    kept_coverage_percent=100*len(kept_gt)/gt['denominator_identities'],
                    matched_query_count=len(linked),
                    additional_queries_on_matched_identities=len(linked)-match['query']['matched_identities'])
        scenes.append(summary)
        assets.extend(selected)
    terminal = all(s['generation_complete'] for s in scenes)
    failed = sum(s['audit_counts']['failed'] for s in scenes)
    report = dict(updated_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        status=('completed_with_failures' if failed else 'completed') if terminal else batch.get('stage', 'pending'),
        queue_finished=terminal, all_queries_succeeded=terminal and not failed,
        source_code_commit='8ab753d0318c994051b7d0d4f72e427427eb37d6',
        scenes=scenes, assets=assets, batch=batch,
        definitions=dict(aor='valid Gaussian export and nonempty 16-view render / all admitted queries',
            bank_acceptance='keep after visual audit, heading and scale checks / all admitted queries',
            coverage='unique annotated identities with a final kept asset / native eligible annotation identities',
            identity_limit='Query IDs are deduplicated candidates, not a proven census of physical scene objects.',
            scope='Only one scene per dataset. Remaining 18 scenes have not been started.'))
    write_json(OUT / 'report.json', report)
    with (OUT / 'asset_inventory.csv').open('w', newline='') as handle:
        fields = ['dataset','scene_id','obj_id','category','route','real_count','generated','render_valid','status','quality_score','error']
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(assets)
    write_page(report)
    write_library_index()
    return report


def write_library_index():
    manifest=LIBRARY/'manifests/asset_library_rigid.json'
    if not manifest.is_file():return
    raw=manifest.read_bytes();data=json.loads(raw);rows=data['assets']
    def link(path,label):
        return f'<a href="{html.escape(os.path.relpath(path,LIBRARY))}">{html.escape(label)}</a>'
    categories={category:[r for r in rows if r['category']==category] for category in sorted({r['category'] for r in rows})}
    counts={category:dict(total=len(items),**{state:sum(r['status']==state for r in items) for state in ['keep','review','reject']})
            for category,items in categories.items()}
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{max-width:1500px;margin:24px auto;padding:0 20px;font:16px/1.6 system-ui;background:#111923;color:#e7eef7}a{color:#8ac6ff}table{width:100%;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #3f5066}img{width:110px;height:80px;object-fit:contain;background:white}.keep{color:#4bd29d}.review{color:#ffd166}.reject{color:#ff8c8c}</style>'
    page+=f'<h1>统一刚体资产库 · {len(rows)} 个审核输出</h1><p>按类别组织，状态来自自动审核。KEEP 为允许检索的资产；REVIEW 与 REJECT 保留供复核，旧 pipeline 的 AssetLibrary 默认不选择它们。这里不保证每个 KEEP 已经人工确认合格或物理身份完全无重复。</p>'
    page+='<p>PLY 坐标：+Z 向上、水平居中、接地，尺度仍为规范化尺度。米制尺寸、尺度可信度、方向、接地方式及来源记录在清单中。使用旧 AssetLibrary 时将本目录作为 pipeline_root。</p><p>'+link(manifest,'完整旧格式清单')+' · '+link(OUT/'START_HERE.html','本轮场景实验与审核结果')+'</p>'
    page+='<table><tr><th>类别</th><th>全部</th><th>自动保留</th><th>待复核</th><th>未通过</th></tr>'
    for category,c in counts.items():
        page+=f'<tr><td><a href="#{category}">{category}</a></td>'+''.join(f'<td>{c[k]}</td>' for k in ['total','keep','review','reject'])+'</tr>'
    page+='</table>'
    for category,items in categories.items():
        page+=f'<details id="{category}"><summary>{category} · {len(items)} 个</summary><table><tr><th>原图</th><th>对象</th><th>状态</th><th>路线</th><th>文件与证据</th></tr>'
        for row in sorted(items,key=lambda r:({'keep':0,'review':1,'reject':2}[r['status']],r['uid'])):
            source=row['provenance'].get('source_image');image=''
            if source:
                image=f'<img loading="lazy" src="{html.escape(os.path.relpath(source,LIBRARY))}">'
            identifier=row['provenance'].get('source_obj_id',row['uid']);evidence=row['quality']['audit_image']
            page+=f'<tr><td>{image}</td><td>{html.escape(identifier)}<br>{html.escape(row.get("dataset",""))}</td><td class="{row["status"]}">{row["status"].upper()}</td><td>{row.get("route","")}</td><td>'+link(LIBRARY/row['canonical']['canonical_path'],'高斯 PLY')+' · '+link(evidence,'原图与16视角审核图')+'</td></tr>'
        page+='</table></details>'
    path=LIBRARY/'rigid_index.html';temporary=path.with_suffix('.tmp');temporary.write_text(page+'</html>');temporary.replace(path)
    import hashlib
    write_json(LIBRARY/'manifests/rigid_category_counts.json',dict(source_manifest_sha256=hashlib.sha256(raw).hexdigest(),counts=counts))


def write_page(report):
    escape = html.escape
    def link(path, label):
        return f'<a href="{escape(os.path.relpath(path, OUT))}">{escape(label)}</a>'
    rows = []
    for s in report['scenes']:
        counts = s['audit_counts']
        values = [s['dataset'],s['scene_id'],s['before_revision_queries'],s['queries'],s['generated'],
            s['render_valid'],counts['keep'],counts['review'],counts['reject'],counts['failed'],counts['pending']]
        rows.append('<tr>'+''.join(f'<td>{v}</td>' for v in values)+'</tr>')
    content = '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    content += '<title>两个场景完整流水线结果</title><style>body{max-width:1500px;margin:24px auto;padding:0 20px;background:#111923;color:#e7eef7;font:16px/1.6 system-ui}a{color:#8ac6ff}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #45546a}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}article{background:#1e2a38;padding:12px;overflow-wrap:anywhere}img{max-width:100%}.source{height:100px}.muted{color:#afbed0}</style>'
    content += f'<h1>Waymo、nuScenes 各一个场景 · 完整生成与筛选</h1><p>更新：{report["updated_utc"]}；状态：{report["status"]}。未完成的对象不计为通过，也不计为质量拒绝。</p>'
    if report['status'] == 'completed_with_failures':
        content += '<p><strong>本轮队列已结束，但存在处理失败，不能视为全部生成成功。质量淘汰与运行失败分别统计。</strong></p>'
    elif report['status'] == 'recovery_running':
        content += '<p><strong>正在补跑显存不足导致的失败项；保留已成功的输出与原审核结论。</strong></p>'
    content += '<p>检测 → 跨帧关联 → SAM 分割与像素质量 → Qwen 完整性/类别审查 → 同物体合并 → FLUX → 车辆 H（0–4 张合格真实图）／其他物体 B0 → 16 视角渲染 → 质量、方向、尺寸审核 → 分类入库。</p>'
    content += '<table><thead><tr>'+''.join(f'<th>{v}</th>' for v in ['数据集','场景','旧版候选','本轮候选','生成成功','16视角有效','自动保留','待复核','未通过','处理失败','进行中'])+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
    content += '<p>候选是当前去重后的生成任务，不能当作场景内真实独立物体总数。自动保留数量是当前规则允许入库的资产数，不等于人工确认全部合格；待复核和拒绝结果保留作诊断，旧版资产加载器会排除它们。</p>'
    if (OUT/'remaining_issues.json').exists():
        content += '<p>阶段预览仍发现一组同源金属盖板被分成 bollard / road_debris 两类并重复生成，说明跨类别重复及类别误判尚未完全解决。本轮继续使用同一套规则完成两场景，残留问题如实保留，不中途挑掉坏例子。'+link(OUT/'remaining_issues.json','查看残留问题记录')+'</p>'
    content += '<p>'+link(OUT/'report.json','完整统计 JSON')+' · '+link(OUT/'asset_inventory.csv','逐资产 CSV')+' · '+link(LIBRARY/'rigid_index.html','按类别浏览统一资产库')+'</p>'
    preview_pointer = OUT / 'latest_preview.json'
    preview = Path(read_json(preview_pointer)['gallery']) if preview_pointer.exists() else OUT / 'kept_previews/START_HERE.html'
    if preview.exists():
        content += '<p>'+link(preview,'查看筛选通过资产的 4×2 环绕视频')+'</p>'
    early=OUT/'first40_preview/START_HERE.html'
    if early.exists():
        content += '<p>'+link(early,'首批40个候选中通过的8个资产预览（非整场景最终结果）')+'</p>'
    full_early=OUT/'all40_preview/START_HERE.html'
    if full_early.exists():
        content += '<p>'+link(full_early,'首批40个完整审核对照视频：包含全部保留、待复核、未通过结果')+'</p>'
    partial_metrics=OUT/'metrics_first40.json'
    if partial_metrics.exists():
        content += '<p>'+link(partial_metrics,'首批40个的阶段指标（完整场景尚未结束）')+'</p>'
    sheet_index=OUT/'contact_sheets_latest.json'
    if sheet_index.exists():
        sheets=read_json(sheet_index)
        content += '<p>'+link(sheets['gallery'],f'查看 {sheets["assets"]} 个已输出资产的图片拼图（含未通过结果）')+' · '+link(sheets['zip'],'下载全部拼图')+'</p>'
    metrics_path=OUT/'metrics.json'
    if metrics_path.exists():
        data=read_json(metrics_path)
        content += f'<p>下面的指标快照包含 {data["protocol"]["exported_payloads"]} 个成功输出；若正在补跑，指标将在本轮补跑结束后更新。</p>'
        content += '<h2>源图一致性指标</h2><p>所有有效输出与最终通过子集分别统计；这两个场景的对象分布不同，不能用此表断言 H 比 B0 更强。每项括号内为有效对象数。FID 是源图与16视角渲染的分布距离，不是插入场景效果分数。</p><table><tr><th>数据集 / 子集</th><th>CLIP-I16 ↑</th><th>DINO-I16 ↑</th><th>FID ↓</th><th>ED-R ↓</th><th>PSNR ↑</th><th>SSIM ↑</th><th>LPIPS ↓</th></tr>'
        for row in data['summary']:
            content += '<tr><td>'+escape(row['dataset']+' / '+row['cohort'])+'</td>'
            for key in ['CLIP_I16','DINO_I16','FID_2048','ED_R','PSNR_dB','SSIM','LPIPS_alex']:
                value=row['metrics'][key];score=value.get('value') if key=='FID_2048' else value['mean']
                count=value.get('n_identities') if key=='FID_2048' else value['n']
                content += '<td>'+('N/A' if score is None else f'{score:.4f} ({count})')+'</td>'
            content += '</tr>'
        content += '</table><p>ED-R 与像素指标仅在原生对象姿态及生成车辆正向都可核验时计算；不按结果搜索最佳视角。无有效样本记 N/A。</p><p>'+link(metrics_path,'指标定义、样本数和逐对象结果')+'</p>'
    for s in report['scenes']:
        merged=s['post_source_removal_reasons'].get('duplicate_source_identity_after_category_verification',0)
        content += f'<h2>{s["dataset"]} / {s["scene_id"]}</h2><p>处理 {s["images"]} 张图；检测框 {s["detections"]}；同相机关联轨迹片段 {s["tracklets"]}。SAM 最多处理 320 个候选身份，本场景实际分割 {s["sam_observations"]} 个观测；像素/掩膜筛选后 {s["mask_pass"]} 个，语义完整性筛选后 {s["source_quality_pass"]} 个，后续质量与去重后 {s["queries"]} 个（其中合并重复候选 {merged} 个）。</p>'
        content += f'<p>技术 AOR：{s["render_valid"]}/{s["queries"]} = {s["aor_percent"]:.2f}%；质量通过率：{s["audit_counts"]["keep"]}/{s["queries"]} = {s["bank_acceptance_percent"]:.2f}%。在生成结束前，两者仅为当前进度值。</p>'
        content += '<table><tr>'+''.join(f'<th>{name}</th>' for name in ['类别','候选','生成成功','自动保留','待复核','未通过','处理失败','进行中'])+'</tr>'
        for category,counts in s['by_category'].items():
            content += '<tr><td>'+escape(category)+'</td>'+''.join(f'<td>{counts[k]}</td>' for k in ['queries','generated','keep','review','reject','failed','pending'])+'</tr>'
        content += '</table>'
        if s.get('coverage'):
            c=s['coverage']; t=c['thresholds']['0.5']
            content += f'<p>原生标注覆盖（IoU≥0.5）：最终通过资产覆盖 {t["kept_identities"]}/{c["denominator"]} 个标注身份；检测覆盖 {t["detector_identities"]}，前端入选覆盖 {t["admitted_identities"]}。匹配到同一标注身份的额外候选有 {t["additional_queries_on_matched_identities"]} 个，此值仅适用于有标注且匹配成功的子集。</p>'
            content += '<p class="muted">nuScenes 标注不覆盖信号灯、杆等所有开放类别；Waymo 无跨相机关联的对象按相机轨迹身份计数。两者分母不同，不能直接把该覆盖率当作全部可用资产召回率。</p>'
        for state in NAMES:
            selected = [a for a in report['assets'] if a['dataset']==s['dataset'] and a['status']==state]
            if not selected:
                continue
            content += f'<details {"open" if state=="keep" else ""}><summary>{NAMES[state]}：{len(selected)}</summary><div class="grid">'
            for a in selected:
                content += f'<article><b>{escape(a["obj_id"])} · {escape(a["category"])} · {a["route"]}</b><p>真实参考图：{a["real_count"]}；质量分：{a["quality_score"]}</p>'
                source=escape(os.path.relpath(a['source_image'],OUT))
                content += f'<a href="{source}"><img class="source" loading="lazy" src="{source}" alt="原始分割图"></a>'
                if a['audit_image']:
                    image=escape(os.path.relpath(a['audit_image'],OUT))
                    content += f'<a href="{image}"><img loading="lazy" src="{image}" alt="原图与16视角实际渲染"></a>'
                reason=a['defects'] or ([a['error']] if a['error'] else [a['review_reason']] if a['review_reason'] else [])
                content += '<p>'+escape('；'.join(reason))+'</p>'
                if a['published']:
                    content += link(a['published']['canonical_path'],'高斯 PLY')
                content += '</article>'
            content += '</div></details>'
    target=OUT/'START_HERE.html'; temp=target.with_suffix('.tmp'); temp.write_text(content+'</html>');temp.replace(target)


def contact_sheets():
    """Arrange unchanged source images and existing Gaussian renders into figures."""
    os.environ.setdefault('MPLCONFIGDIR','/tmp/asset_report_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    import zipfile

    report=snapshot()
    library={r['provenance']['source_obj_id']:r for r in
        read_json(LIBRARY/'manifests/asset_library_rigid.json')['assets']
        if r['provenance'].get('batch_id')=='batch20_20261024:frontend_revision_01'}
    records=[]
    for row in report['assets']:
        if not row['published']:
            continue
        entry=library[row['obj_id']]
        evidence=entry['quality']['rendering']
        records.append(dict(index=len(records)+1,**row,render_sheet=evidence['evidence'],
            render_resolution=evidence['resolution'],render_sheet_sha256=sha256(evidence['evidence'])))
    if not records:
        raise ValueError('No published rendered assets yet')
    output=OUT/f'asset_sheets_{len(records)}';output.mkdir(exist_ok=True)
    colors={'keep':'#4bd29d','reject':'#ff8c8c','review':'#ffd166'}
    groups=[(f'all_{i//8+1:02d}',records[i:i+8],f'ALL OUTPUTS  |  {i+1:03d}-{min(i+8,len(records)):03d} / {len(records)}')
            for i in range(0,len(records),8)]
    kept=[r for r in records if r['status']=='keep']
    groups += [(f'kept_{i//8+1:02d}',kept[i:i+8],f'AUTOMATIC KEEP  |  {i+1}-{min(i+8,len(kept))} / {len(kept)}')
               for i in range(0,len(kept),8)]
    figure_records=[]
    for name,rows,group_title in groups:
        fig=plt.figure(figsize=(24,16),dpi=100,facecolor='#111923')
        fig.text(.025,.972,group_title,color='white',fontsize=22,weight='bold')
        fig.text(.025,.944,'Source + 4 actual Gaussian views (45 / 135 / 225 / 315 deg)   |   KEEP = automatic pass; REVIEW = uncertain; REJECT = automatic fail',
            color='#becbdb',fontsize=12)
        grid=fig.add_gridspec(4,2,left=.02,right=.98,bottom=.025,top=.915,wspace=.045,hspace=.18)
        for pos,row in enumerate(rows):
            cell=grid[pos//2,pos%2];box=cell.get_position(fig)
            title=f'{row["index"]:03d}  {row["category"]}  |  {row["route"]}'
            if row['route']=='H':title+=f'  real={row["real_count"]}'
            fig.text(box.x0,box.y1+.008,title,color='white',fontsize=15,weight='bold')
            fig.text(box.x1,box.y1+.008,row['status'].upper(),color=colors[row['status']],fontsize=15,weight='bold',ha='right')
            fig.text(box.x0,box.y0-.008,row['obj_id'],color='#bac9da',fontsize=11,va='top')
            fig.text(box.x1,box.y0-.008,f'Qwen score: {row["quality_score"]}/10',color='#bac9da',fontsize=11,va='top',ha='right')
            panels=cell.subgridspec(1,5,wspace=.025)
            source=Image.open(row['source_image']).convert('RGB')
            sheet=Image.open(row['render_sheet']).convert('RGB');size=row['render_resolution']
            if sheet.size!=(size*4,(size+24)*4):
                raise ValueError('Unexpected render sheet geometry')
            images=[source]
            for view in [2,6,10,14]:
                x=view%4*size;y=view//4*(size+24)+24
                images.append(sheet.crop((x,y,x+size,y+size)))
            for panel,(im,label) in enumerate(zip(images,['SOURCE','45 deg','135 deg','225 deg','315 deg'])):
                ax=fig.add_subplot(panels[0,panel]);ax.imshow(im);ax.set_axis_off()
                ax.set_title(label,color='#adc1d6',fontsize=9,pad=5)
        path=output/f'{name}.jpg'
        fig.savefig(path,dpi=100,facecolor=fig.get_facecolor(),pil_kwargs={'quality':94,'subsampling':0})
        plt.close(fig)
        figure_records.append(dict(file=path.name,title=group_title,obj_ids=[r['obj_id'] for r in rows],
            count=len(rows),sha256=sha256(path)))
    manifest=dict(assets=len(records),figures=figure_records,source='Existing source crops and actual 3DGS render evidence; no synthesized replacement pictures',
        angles=[45,135,225,315],snapshot=report['updated_utc'],status_counts=dict(Counter(r['status'] for r in records)),records=records)
    write_json(output/'manifest.json',manifest)
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{max-width:1600px;margin:24px auto;padding:0 20px;background:#111923;color:#eef3fa;font:16px/1.6 system-ui}img{width:100%}a{color:#8ac6ff}article{margin:32px 0}</style>'
    page+=f'<h1>{len(records)} 个已输出资产 · 图片拼图</h1><p>截至 {report["updated_utc"]}，每张最多8个对象，全部输出按原始清单顺序展示。每个对象包含原始分割裁图及45°、135°、225°、315°四个实际高斯渲染视角。各对象独立取景，画面尺寸不代表实际尺寸。</p>'
    page+='<p>绿色 KEEP：自动保留；黄色 REVIEW：待复核；红色 REJECT：自动未通过。审核状态来自模型，不能替代人工确认。末尾另附自动保留子集，前面的 ALL OUTPUTS 已包含所有状态。</p>'
    progress=('两个场景的队列已结束，本页展示所有成功导出的资产；处理失败单列在汇总页。'
              if report.get('queue_finished') else '本轮两个场景正在处理，这只是已完成部分的快照。')
    page+=f'<p>{progress}<a href="../START_HERE.html">返回完整进度</a></p>'
    for record in figure_records:
        label=('自动保留子集' if record['file'].startswith('kept') else '全部输出')+' · '+record['file']
        page+=f'<article><h2>{label}</h2><a href="{record["file"]}"><img loading="lazy" src="{record["file"]}"></a></article>'
    (output/'START_HERE.html').write_text(page+'</html>')
    archive=OUT/f'{output.name}.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.iterdir()):
            if path.is_file():bundle.write(path,arcname=f'{output.name}/{path.name}')
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:raise ValueError('Image archive failed CRC validation')
    write_json(OUT/'contact_sheets_latest.json',dict(assets=len(records),figures=len(figure_records),gallery=str(output/'START_HERE.html'),zip=str(archive),zip_bytes=archive.stat().st_size))
    snapshot()
    return output


def verify_publication(report):
    """Check complete receipts and loading through the original insertion adapter."""
    import numpy as np
    sys.path.insert(0,'/mnt/wmj/workspace4/pipeline/traj_pipeline')
    from corner_case.assets import AssetLibrary,bind_rigid
    central=LIBRARY/'manifests/asset_library_rigid.json'
    rows={r['uid']:r for r in read_json(central)['assets']}
    selected={r['published']['uid']:r for r in report['assets'] if r['published']}
    for uid,item in selected.items():
        entry=rows[uid];payload=entry['canonical']
        if (sha256(LIBRARY/payload['canonical_path'])!=payload['sha256'] or
            payload['sha256']!=item['published']['sha256'] or entry['status']!=item['status']):
            raise ValueError('Publication no longer matches its curation record')
    expected_keep={uid for uid,item in selected.items() if item['status']=='keep'}
    library=AssetLibrary(central,LIBRARY)
    loaded=[e for e in library.entries if e.uid in selected]
    if {e.uid for e in loaded}!=expected_keep:
        raise ValueError('Original loader selected an unexpected quality subset')
    bound=[]
    for entry in loaded:
        asset=bind_rigid(entry,np.random.default_rng(42))
        if not asset.asset_path.is_file() or asset.scale<=0 or min(asset.size_lwh_m)<=0:
            raise ValueError('Original insertion binding failed')
        bound.append(dict(uid=entry.uid,category=entry.category,path=str(asset.asset_path),
                          scale=asset.scale,size_lwh=list(asset.size_lwh_m)))
    completed=[]
    for path in sorted((N/'chunks').glob('*/completed.json')):
        receipt=read_json(path);chunk=path.parent
        if (receipt['generation_sha256']!=sha256(chunk/'generation/generation.json') or
            receipt['curation_sha256']!=sha256(chunk/'curation/curation.json')):
            raise ValueError('Completed chunk records changed')
        completed.append(dict(chunk=chunk.name,queries=receipt['queries']))
    recovery_receipts=[]
    for path in sorted((N/'chunks').glob('*/recovery_*/completed.json')):
        receipt=read_json(path);work=path.parent
        if receipt['generation_sha256']!=sha256(work/'generation/generation.json') or receipt['curation_sha256']!=sha256(work/'curation/curation.json'):
            raise ValueError('Recovery records changed')
        recovery_receipts.append(dict(attempt=str(work),queries=receipt['queries']))
    if report.get('queue_finished') and (sum(r['queries'] for r in completed)!=len(report['assets']) or
        any(r['status']=='pending' for r in report['assets'])):
        raise ValueError('Full-scene completion is missing query outcomes')
    result=dict(published=len(selected),original_loader_kept=len(loaded),bound=bound,
        receipts=completed,recovery_receipts=recovery_receipts,all_payload_hashes_verified=True,
        quality_scope='Automatic keep, not a claim that manual inspection found no defects',
        insertion_scope='Original AssetLibrary loading and bind_rigid integration; not a rendered scene insertion experiment')
    write_json(OUT/'publication_verification.json',result)
    return result


async def finalize_previews(report):
    groups, index = [], 1
    for dataset in SCENES:
        rows=[]
        for item in sorted(report['assets'],key=lambda a:(a['category'],a['obj_id'])):
            if item['dataset'] != dataset or item['status'] != 'keep':
                continue
            payload=item['published']
            if not payload or sha256(payload['canonical_path']) != payload['sha256']:
                raise ValueError('Kept asset is not verifiably published')
            rows.append(dict(index=index,obj_id=item['obj_id'],category=item['category'],scene_id=item['scene_id'],
                route=item['route'],real_count=item['real_count'],raw_path=payload['canonical_path'],
                raw_sha256=payload['sha256'],source_image=item['source_image'],source_sha256=item['source_sha256'],
                audit_status='keep',audit_error=None,defects=item['defects'],payload_is_canonical=True,canonical=item['canonical']))
            index+=1
        groups.extend(rows[start:start+8] for start in range(0,len(rows),8))
    if not groups:
        write_json(OUT/'preview_status.json',dict(status='no_kept_assets',count=0));return
    manifest=write_json(OUT/'kept_preview_manifest.json',dict(scope='curated_scene_pilots',asset_count=index-1,
        groups=groups,frames=128,fps=16,duration_seconds=8,resolution=[2048,1152],layout='4 columns x 2 rows'))
    preview_dir=OUT/os.environ.get('ASSET_PILOT_PREVIEW_DIR','kept_previews')
    env=dict(os.environ,ASSET_PREVIEW_MANIFEST=str(manifest),ASSET_PREVIEW_OUTPUT=str(preview_dir),
             ASSET_PREVIEW_GPUS=os.environ.get('ASSET_PREVIEW_GPUS','3,4,5,6,7'))
    with (OUT/'preview.log').open('a') as log:
        process=await asyncio.create_subprocess_exec(sys.executable,str(B/'scripts/preview224.py'),env=env,
            stdout=log,stderr=asyncio.subprocess.STDOUT)
        if await process.wait():
            raise RuntimeError('Final kept preview failed; inspect preview.log')
    write_json(OUT/'latest_preview.json',dict(gallery=str(preview_dir/'START_HERE.html'),assets=index-1))
    snapshot()


async def watch():
    while True:
        report=snapshot()
        if report['status'] in {'completed','completed_with_failures'}:
            contact_sheets()
            await finalize_previews(report)
            write_json(OUT/'delivery_status.json',dict(status='generation_screening_preview_completed',
                kept=sum(s['audit_counts']['keep'] for s in report['scenes']),
                metric_status='feature_metrics_pending',updated=time.time()))
            with (OUT/'metrics_pipeline.log').open('a') as log:
                process=await asyncio.create_subprocess_exec(sys.executable,str(B/'scripts/evaluate_batch.py'),'pipeline',
                    stdout=log,stderr=asyncio.subprocess.STDOUT)
                if await process.wait():
                    raise RuntimeError('Pilot feature metrics failed; inspect metrics_pipeline.log')
            final=snapshot()
            verify_publication(final)
            write_json(OUT/'delivery_status.json',dict(status=('two_scene_pilots_completed' if final['all_queries_succeeded'] else 'two_scene_pilots_finished_with_failures'),
                kept=sum(s['audit_counts']['keep'] for s in report['scenes']),metric_status='completed',updated=time.time()))
            return
        if report['status']=='failed':
            raise RuntimeError('Generation stopped; report preserves all current counts')
        await asyncio.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');parser.add_argument('--sheets',action='store_true');args=parser.parse_args()
    if args.watch:asyncio.run(watch())
    elif args.sheets:print(contact_sheets())
    else:
        report=snapshot()
        print(json.dumps(dict(status=report['status'],scenes=report['scenes']),ensure_ascii=False))
