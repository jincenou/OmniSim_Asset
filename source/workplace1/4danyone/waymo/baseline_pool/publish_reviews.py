"""Compact navigation for existing manual input reviews; no media copies."""
import html
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = {
    'candidate': '已保留的输入候选',
    'completion': '待补全／仍有不确定性的片段',
    'hold': '当前窗口暂缓，保留身份与失败证据',
    'partial': '复核范围不足／仍待判断',
}


def category(review):
    if review.get('input_review_passed') is True:
        return 'candidate'
    state = review.get('state', '')
    if state.startswith('retain_completion_candidate'):
        return 'completion'
    if state.startswith('hold_window') or state in {'retry_other_window', 'retry_shorter_window'}:
        return 'hold'
    return 'partial'


def publish(paths, css=''):
    groups = defaultdict(list)
    for ready_path in paths:
        review_path = ready_path.parent / 'review.json'
        if not review_path.is_file():
            continue
        review = json.loads(review_path.read_text())
        data = json.loads(ready_path.read_text())
        row = data['candidate']
        groups[category(review)].append((ready_path, data, review))
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    count = sum(map(len, groups.values()))
    document = [css, '<h1>Waymo 输入与骨架：已复核片段</h1>',
                '<p><a href="index.html">全部场景输入池</a> · <a href="REPORT.md">详细报告</a> · '
                '<a href="generation227/index.html">独立的227条生成结果</a></p>',
                f'<p>更新：{timestamp}。这里只列有人工复核记录的 {count} 个准备包；'
                f'全部准备包快照为 {len(paths)} 个。以下均为窗口数，不是不同人数。</p>',
                '<p>输入候选不等于高质量生成资产；没有新运行多视角RGB生成。'
                '局部缺失或短暂遮挡不直接淘汰身份。接触表逐帧查看也不是实时播放或真实3D动作验收。'
                '每条的限制以复核记录为准，未复核对象仍在总输入池。</p><ul>']
    for key, label in LABELS.items():
        document.append(f'<li><a href="#{key}">{label}</a>：{len(groups[key])} 个窗口</li>')
    document.append('</ul>')

    def link(path):
        path = Path(path)
        assert path.is_file(), path
        return html.escape(os.path.relpath(path, ROOT), quote=True)

    for key, label in LABELS.items():
        document.append(f'<h2 id="{key}">{label}</h2>')
        for ready_path, data, review in sorted(groups[key], key=lambda x: str(x[0])):
            folder = ready_path.parent
            c = data['candidate']
            identity = html.escape(c['identity_scope'])
            title = html.escape(f"{c['split']}/{c['scene']} · {c['review_id']} · camera {c['camera_index']}")
            alternative = '同人替代窗口，不新增身份' if folder.is_relative_to(ROOT/'retry') else '主窗口'
            document.append(f'<section data-category="{key}"><h3>{title}</h3>'
                            f'<p>{alternative}；{c["native_frame_count"]} 原生帧 → {c["output_slots"]} 输入槽。身份依据：{identity}。</p>'
                            f'<p>{html.escape(review.get("notes", "见复核记录。"))}</p><div class="videos">')
            for source, name in [(folder/'clip0/observed_native.mp4', '原生观测'),
                                 (folder/data['inputs'], '实际输入'),
                                 (folder/data['input_skeleton_comparison'], '输入＋GVHMR骨架')]:
                href = link(source)
                document.append(f'<div><p>{name} · <a href="{href}">打开视频</a></p>'
                                f'<video controls preload="none" src="{href}"></video></div>')
            document.append(f'</div><p><a href="{link(folder/"review.json")}">具体复核范围</a> · '
                            f'<a href="{link(ready_path)}">输入包清单</a></p></section>')
    target = ROOT/'reviewed_inputs.html'
    temporary = target.with_suffix(f'.html.{os.getpid()}.tmp')
    temporary.write_text('\n'.join(document))
    temporary.replace(target)
    return {key:len(groups[key]) for key in LABELS}


if __name__=='__main__':
    from package_catalog import ready_paths
    from publish_pool import CSS
    print(json.dumps(publish(ready_paths(),CSS),ensure_ascii=False))
