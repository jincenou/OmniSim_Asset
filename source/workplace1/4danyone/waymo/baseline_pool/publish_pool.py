"""Incremental index of complete input packages only, no generated results."""
import argparse
import asyncio
import collections
import html
import json
import os
from pathlib import Path
import time

from scan_native import ROOT, write
from package_catalog import ready_paths
from pool_accounting import summarize
from publish_reviews import publish as publish_reviews

CSS = """<meta charset="utf-8"><style>body{font:16px sans-serif;background:#17191d;color:#eee;margin:24px}a{color:#9bd}section{background:#252830;padding:18px;margin:18px 0}.videos{display:flex;gap:12px}.videos>div{flex:1;min-width:0}video{width:100%;max-height:450px}p{line-height:1.6}table{border-collapse:collapse}td,th{border:1px solid #555;padding:8px}</style>"""


def publish():
    groups = collections.defaultdict(list)
    all_rows = []
    paths = ready_paths()
    for path in paths:
        data = json.loads(path.read_text())
        assert data["generation_started"] is False
        row = data["candidate"]
        for key in ["inputs", "input_skeleton_comparison", "motion", "conditioning"]:
            assert (path.parent / data[key]).exists()
        groups[row["split"], row["scene"]].append((path, data))
        all_rows.append({"split": row["split"], "scene": row["scene"], "identity_key": row["identity_key"],
                         "identity_scope": row["identity_scope"], "ready": str(path),
                         "alternative_window": path.is_relative_to(ROOT / "retry"),
                         "quality_accepted": False, "state": data["state"]})
    gallery = ROOT / "gallery"
    gallery.mkdir(exist_ok=True)
    table = []
    for (split, scene), items in sorted(groups.items()):
        name = f"{split}_{scene}.html"
        blocks = []
        for path, data in items:
            row = data["candidate"]
            directory = path.parent
            sources = [(directory / "clip0/observed_native.mp4", "原生观测"),
                       (directory / data["inputs"], "实际输入"),
                       (directory / data["input_skeleton_comparison"], "输入＋GVHMR投影")]
            assert all(p.is_file() for p, _ in sources)
            players = "".join(
                f'<div><p>{title}</p><video controls preload="none" src="{html.escape(os.path.relpath(p, gallery))}"></video></div>'
                for p, title in sources)
            review_path = directory / "review.json"
            review = json.loads(review_path.read_text()) if review_path.exists() else {}
            audit_path = directory / "skeleton_audit.json"
            audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
            notes = html.escape(review.get("notes", "尚无人工复核结论。"))
            flags = html.escape("；".join(audit.get("flags", [])) or "无已记录数值告警；不代表质量通过")
            alternative = " · 同人替换窗口，不新增人数" if path.is_relative_to(ROOT / "retry") else ""
            blocks.append(f'<section><h2>{html.escape(row["review_id"])} · camera {row["camera_index"]}</h2>'
                          f'<p>身份依据：{html.escape(row["identity_scope"])}；{row["output_slots"]}帧。'
                          f'{alternative} 准备完成，最终质量未验收；局部遮挡不自动排除。</p>'
                          f'<p>人工抽查：{notes}</p><p>数值复核提示：{flags}</p>'
                          f'<div class="videos">{players}</div>'
                          f'<p><a href="{html.escape(os.path.relpath(path, gallery))}">输入包清单与骨架条件路径</a></p></section>')
        page = CSS + f'<h1>{split}/{scene} 输入与骨架</h1><p><a href="../index.html">返回总览</a></p>' + "".join(blocks)
        temporary = gallery / (name + ".writing")
        temporary.write_text(page)
        temporary.replace(gallery / name)
        table.append(f'<tr><td><a href="gallery/{name}">{split}/{scene}</a></td><td>{len(items)}</td></tr>')
    native = json.loads((ROOT / "native_summary.json").read_text())
    predicted = json.loads((ROOT / "predicted_summary.json").read_text())
    statuses = {}
    for filename in ["native_prepare_status.json", "predicted_prepare_status.json", "processed_pilot_status.json", "retry_prepare_status.json", "retry2_prepare_status.json", "crop_retry_prepare_status.json", "foreground_retry_prepare_status.json", "alternative_retry_prepare_status.json", "native_mask_retry_prepare_status.json", "terminal_retry_prepare_status.json", "review7_boundary_retry_prepare_status.json", "review7_native_boundary_retry_prepare_status.json", "review8_predicted_retry_prepare_status.json", "review8_native_retry_prepare_status.json", "review9_predicted_retry_prepare_status.json", "review9_native_retry_prepare_status.json"]:
        path = ROOT / filename
        if path.exists():
            data = json.loads(path.read_text())
            statuses[filename] = dict(collections.Counter(v["state"] for v in data["jobs"].values()))
    summary = {"updated_at": time.time(), "scanned_scenes": native["scene_count"] + predicted["scene_count"],
               "native_candidate_keys": native["deduplicated_candidate_keys"],
               "predicted_camera_tracks": predicted["candidate_camera_tracks"],
               "prepared_packages": len(all_rows), "quality_accepted": None,
               "alternative_window_packages": sum(row["alternative_window"] for row in all_rows),
               "verified_unique_people_total": None, "preparation_status": statuses,
               "scope": "Local1000scenes; camera candidates are not deduplicated people; quality review pending."}
    write(ROOT / "POOL_STATUS.json", summary)
    write(ROOT / "prepared_index.json", {"packages": all_rows})
    accounting = summarize(paths)
    publish_reviews(paths, CSS)
    main_queues_finished = all(
        json.loads((ROOT / name).read_text()).get("state") == "finished_requires_review"
        for name in ("native_prepare_status.json", "predicted_prepare_status.json")
    )
    preparation_label = ("输入池主准备队列已结束，质量复核与身份关联尚未完成"
                         if main_queues_finished else "输入池继续全量准备")
    document = CSS + f"""<h1>Waymo 基线输入池</h1>
<p><a href="generation227/index.html"><strong>新授权批次：227条候选的4DAnyone多视角效果与进度</strong></a>。
{preparation_label}；227条生成已完成，分类与问题诊断见上方入口。下面的READY和计数只描述输入准备，不能据此判断生成质量。</p>
<p>已扫描 {summary['scanned_scenes']} 个本地场景；原生标注候选键 {summary['native_candidate_keys']}，
预测相机轨迹 {summary['predicted_camera_tracks']}。这些数量不可相加当作独立人数。</p>
<p>当前完整准备包：{len(all_rows)}。以下场景页展示实际输入与骨架；新多视角RGB结果见上方独立入口。
其中 {summary['alternative_window_packages']} 包是同一身份的替换窗口，不增加独立人数。
局部遮挡或缺失肢体不直接排除，需看GVHMR的姿态和时序。自动准备完成不等于质量合格。</p>
<p>完整包涉及官方关联的场景内身份键 {accounting['prepared_official_scene_identity_keys']} 个；
其余相机轨迹身份待关联，不能相加当独立人数。最终质量通过人数尚未确定。
已有 {accounting['input_review_candidate_windows']} 个窗口通过有记录的输入复核、保留为生成前候选；这不等于生成后高质量资产。
历史失败尝试 {accounting['historical_failed_attempts']} 次，其中 {accounting['historical_failures_subsequently_prepared_same_window']} 次原窗口后来已完成准备；失败次数不代表淘汰人数。</p>
<p><a href="PLAN.md">筛选规则</a> · <a href="ACCOUNTING.json">身份／窗口分母</a> · <a href="failure_inventory.json">失败原因</a> · <a href="prepared_windows.csv">窗口CSV</a> · <a href="POOL_STATUS.json">当前统计</a> · <a href="prepared_index.json">完整输入包索引</a> · <a href="pilot_view.html">四个工程样本的固定图复核说明</a></p>
<p><strong><a href="reviewed_inputs.html">直接查看已复核输入：候选／待补全／问题窗口</a></strong> · <a href="scene_progress.csv">1000场景进度快照</a> · <a href="input_risk_inventory.csv">输入风险快照</a>（后两项按文件内时间解读）</p>
<table><tr><th>场景</th><th>完整输入包数</th></tr>{''.join(table)}</table>"""
    (ROOT / "index.html.writing").write_text(document)
    (ROOT / "index.html.writing").replace(ROOT / "index.html")
    return summary


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch-pid", nargs="*", type=int, default=[])
    args = parser.parse_args()
    pilot = ROOT / "pilot_view.html"
    if not pilot.exists():
        old = (ROOT / "index.html").read_text()
        assert "四个工程试运行样本" in old
        pilot.write_text(old)
    while True:
        result = publish()
        print(json.dumps(result), flush=True)
        live = False
        for pid in args.watch_pid:
            try:
                os.kill(pid, 0)
                command = Path(f"/proc/{pid}/cmdline").read_bytes()
                if any(name in command for name in [b"run_native_pool.py", b"run_predicted_pool.py", b"run_processed_pilot.py", b"run_crop_retry_pool.py", b"run_foreground_retry_pool.py", b"run_alternative_retry_pool.py", b"run_native_mask_retry_pool.py"]):
                    live = True
            except (ProcessLookupError, FileNotFoundError):
                pass
        # Retention checks can finish just after their parent preparation queue.
        # Publish their final READY files before ending this watcher.
        if not live:
            for proc in Path('/proc').iterdir():
                if not proc.name.isdigit():
                    continue
                try:
                    argv = (proc / 'cmdline').read_bytes().split(b'\0')
                    if any(arg in {os.fsencode(ROOT / 'watch_retention.py'),
                                   os.fsencode(ROOT / 'repair_empty_retention.py')}
                           for arg in argv):
                        live = True
                        break
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    continue
        if not live:
            break
        await asyncio.sleep(45)


if __name__ == "__main__":
    asyncio.run(main())
