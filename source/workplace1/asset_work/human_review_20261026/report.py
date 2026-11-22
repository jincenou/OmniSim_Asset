"""Refresh audit summaries; never write to the published library."""
import csv
import html
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_text())


def publish():
    protocol = read(ROOT/'protocol.json')
    state = read(ROOT/'status.json')
    manual = {x['identity']: x for x in read(ROOT/'manual_review.json')['entries']}
    rows = []
    for entry in protocol['packages']:
        identity = entry['identity']
        asset = Path(protocol['assets'])/identity
        metrics = read(asset/'quality.json')['mean']
        insertion = read(asset/'insertion.json')
        placement = read(asset/'placement.json')
        tech = ROOT/'technical'/f'{identity}.json'
        visual = ROOT/'vlm'/f'{identity}.json'
        error = ROOT/'errors'/f'{identity}.json'
        model = read(visual)['result'] if visual.exists() else {}
        rows.append(dict(identity=identity,
            integrity='passed' if tech.exists() else 'pending_or_error',
            model_triage=model.get('triage','pending_or_error'),
            model_note=model.get('evidence',''),
            direct_review=manual.get(identity,{}).get('status','not_reviewed'),
            direct_note=manual.get(identity,{}).get('note','')+' '+manual.get(identity,{}).get('same_view_followup',{}).get('note',''),
            error=read(error)['error'] if error.exists() else '',
            action_estimate=placement['action']['label'],
            skip_insertion=insertion.get('skip_insertion',False),
            ground_verified=placement['feet']['ground_verified'],
            loopable=placement['playback']['loopable'],
            **metrics))
    tech_count=sum(r['integrity']=='passed' for r in rows)
    model_counts=Counter(r['model_triage'] for r in rows)
    means={k:sum(r[k] for r in rows)/len(rows) for k in ['full_psnr','foreground_psnr','ssim','crop_lpips','silhouette_iou','outside_alpha_mean']}
    snapshot=dict(updated_utc=datetime.now(timezone.utc).isoformat(),state=state['state'],
        frozen_assets=len(rows),technical_passed=tech_count,model_counts=dict(model_counts),
        direct_sample_review=len(manual),model_is_acceptance=False,library_modified=False,
        excluded=[r['identity'] for r in rows if r['skip_insertion']],mean_generated_reference_metrics=means)
    (ROOT/'summary.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
    with (ROOT/'inventory.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['已发布人体资产复核（隔离输出）',snapshot['updated_utc'],
        f"冻结范围：{len(rows)} 个已发布资产；任务状态：{state['state']}。",
        f"文件、动态资产哈希、保存的回放记录、坐标/运动侧车契约及全帧视频解码：{tech_count}/{len(rows)} 通过。",
        '这不是重新运行全部 GPU 回放，也不是视觉质量通过。',
        f'VLM 五时刻抽样初筛：{dict(model_counts)}。仅作线索，已发现漏检和误报，不能作为合格数量。',
        f'Codex 直接查看抽样图：{len(manual)} 个；详细范围与证据哈希见 manual_review.json。',
        '正式库未被修改，未删除任何资产。当前生产中新完成的资产不混入本次冻结分母。',
        '原有禁止插入标记：'+', '.join(snapshot['excluded']),
        f"现有元数据 ground_verified=True：{sum(r['ground_verified'] for r in rows)}；loopable=True：{sum(r['loopable'] for r in rows)}。近似信息存在，不等于接地、防滑步、循环已验收。",
        '', '已有重建指标的全库算术均值（逐资产等权，不重算评测）：',
        json.dumps(means,ensure_ascii=False),
        '参考是 4DAnyone 生成图，不是真实 Waymo 新视角。均值不能掩盖携带物丢失或局部肢体缺陷。',
        '', '直接复核记录：']
    lines += [f"{k} [{v['status']}] {v['note']} {v.get('same_view_followup',{}).get('note','')}" for k,v in manual.items()]
    lines += ['', '后续：完成全库自动初筛，优先核查问题候选与重建指标较差者；VLM未报错对象仍不自动放行。',
        '查看 index.html；原始 demo 直接链接正式库，未复制视频。这里只保存低体积抽样证据和报告。']
    (ROOT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    esc=lambda s:html.escape(str(s))
    body=['<!doctype html><meta charset="utf-8"><title>人体资产复核</title>',
        '<style>body{font:16px sans-serif;margin:24px;background:#fafafa}td,th{padding:8px;border:1px solid #ccc;vertical-align:top}table{border-collapse:collapse;width:100%}pre{white-space:pre-wrap}input{padding:10px;width:50%}</style>',
        '<h1>人体资产复核：技术通过不等于视觉通过</h1><pre>'+esc('\n'.join(lines[:11]))+'</pre>',
        '<p><a href="REPORT.txt">报告</a> · <a href="inventory.csv">逐资产清单</a> · <a href="manual_review.json">直接视觉核查证据</a> · <a href="sideview_diagnosis.jpg">生成与重建同视角对照</a></p>',
        '<input placeholder="筛选编号、问题文字或状态" oninput="document.querySelectorAll(\'tbody tr\').forEach(r=>r.hidden=!r.innerText.toLowerCase().includes(this.value.toLowerCase()))">',
        '<table><thead><tr><th>资产与原视频</th><th>技术</th><th>直接抽样核查</th><th>VLM线索（可能误报/漏报）</th><th>前景PSNR / LPIPS</th></tr></thead><tbody>']
    priority={'confirmed_visible_issue':0,'model_flag_not_confirmed':1,'sampled_no_gross_defect':2,'not_reviewed':3}
    for r in sorted(rows,key=lambda r:(priority[r['direct_review']],r['foreground_psnr'])):
        asset=Path(protocol['assets'])/r['identity']
        # Relative links work when browsing the shared filesystem, without copying videos.
        import os
        video=os.path.relpath(asset/'demo.mp4',ROOT)
        contact=f"contacts/{r['identity']}.jpg"
        body.append(f"<tr><td>{esc(r['identity'])}<br><a href='{esc(video)}'>五列视频</a> · <a href='{contact}'>五时刻</a><br>skip={r['skip_insertion']}</td><td>{esc(r['integrity'])}</td><td>{esc(r['direct_review'])}<br>{esc(r['direct_note'])}</td><td>{esc(r['model_triage'])}<br>{esc(r['model_note'])}</td><td>{r['foreground_psnr']:.3f} / {r['crop_lpips']:.4f}</td></tr>")
    body.append('</tbody></table>')
    (ROOT/'index.html').write_text('\n'.join(body))
    return state['state']


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while True:
        status=publish()
        if not args.watch or status!='running':break
        time.sleep(30)
