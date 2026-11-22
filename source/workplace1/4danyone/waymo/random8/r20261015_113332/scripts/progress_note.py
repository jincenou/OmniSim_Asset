"""One stable, short progress note for the complete frozen eight-person matrix."""
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

R = Path(__file__).resolve().parents[1]
BRANCHES = ("ftgs_base5k", "ftgs_opt15k", "4dgs_grid25", "4dgs_grid61")
LABELS = ("N01", "N02", "N03", "N04", "N06", "N07", "N08", "N11")


def write_note(live):
    counts = []
    for label in LABELS:
        models = []
        evaluations = []
        for branch in BRANCHES:
            model = (R / "samples" / label / branch / "training_complete.json" if branch.startswith("ftgs")
                     else R / "four_dg/runs" / f"{label}_grid{branch[-2:]}" / "complete.json")
            if model.exists():
                models.append(branch)
            if (R / "samples" / label / "evaluations" / branch / "results.json").exists():
                evaluations.append(branch)
        counts.append({"label": label, "trained": models, "evaluated": evaluations})
    audit_path = R / "research/matrix_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    accepted = []
    for label, row in audit.get("tracks", {}).items():
        checks = [row[key] for key in ("shared_inputs", "pair", "comparison", "camera_projection")]
        for branch in row["branches"].values():
            checks.extend([branch["training"], branch["final_evaluation"], *branch["budget_evaluations"].values()])
        if all(check["state"] == "passed" for check in checks):accepted.append(label)
    jobs = [row for row in live["jobs"] if row["observed_process_alive"] and row["receipt_state"] == "running"]
    complete = len(accepted) == 8 and audit.get("counts") == {"passed": 132, "pending": 0, "failed": 0}
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    lines = ["# Random8 最新进度", "", f"更新时间：{stamp}（北京时间）。",
             "", "目标：8 条不同的新 Waymo 人体轨迹 × 4 个冻结后端配方，共 32 个重建版本。",
             f"已训练完成 {sum(len(row['trained']) for row in counts)}/32；终态独立重放与主指标已完成 {sum(len(row['evaluated']) for row in counts)}/32。这两个计数不包含中间预算快照。",
             "", "八人的 24 路官方 4DAnyone-Turbo 视频均已生成验收。N05、N09、N10 前端硬失败保留；总计尝试 11 条，未额外增加目标人数。",
             "", "|人物|终态训练 / 4|主评测 / 4|", "|---|---:|---:|"]
    lines.extend(f"|{row['label']}|{len(row['trained'])}|{len(row['evaluated'])}|" for row in counts)
    if audit:
        lines.extend(["", f"完整交付技术检查已通过 {len(accepted)}/8 人：{', '.join(accepted) or '尚无'}。检查包括共同输入、训练/预算/主评测、动态文件、完整视频、两网格配对和相机映射；它不等于人工外观质量批准。", f"矩阵检查计数：{audit.get('counts', {})}，更新时间 {audit.get('updated_at')}。详见 [matrix_audit.json](research/matrix_audit.json)。"])
    lines.extend(["", "最新授权：物理 GPU 0–7。所有队列共用设备锁、按实际显存准入，并校验实际 CUDA UUID。允许与其他项目共享，保留正在运行的计算。",
                  "", "现场存活任务：", ""])
    lines.extend(f"- `{row['job']}`：GPU {row.get('physical_gpus', [])}，PID {row['pid']}；日志 `{Path(row['log']).name}`。" for row in jobs)
    if not jobs:lines.append("本轮GPU任务均已结束。" if complete else "当前未观察到存活的本轮GPU任务，终态以交付检查为准。")
    lines.extend(["", "后台队列均已结束，原tmux任务自动退出，无需重启。历史命令、PID、UUID和退出码保留在状态/日志中。" if complete else "后台：`tmux -L fdwaymo attach -t random8_r20261015_113332`。",
                  "", "本轮共同输入、四组冷启动优化、独立重放、三个留出视角完整时序/六项指标及完整五栏对照均已完成；终态与中间预算共64行指标。" if complete else "后续自动接续：共同 RGBA/初始化 → 四组从头优化 → 独立动态包重放 → 完整五栏对照及汇总。",
                  "", "仅四组全部评测通过才发布对照视频和局部对照图；不生成 PENDING 栏、不保留 previous 副本。局部图固定在 `comparison/details/`，预算快照仅保留数值。SAM2 临时 JPEG 在分割成功退出后清理。模型、恢复 checkpoint、共享 RGBA、原始视频、有效最终展示和失败日志保留。",
                  "", "指标为 generated-heldout reconstruction，不是真实 Waymo 新视角精度。世界尺度、地面接触、任意换动作及场景融合仍未验证，人工质量批准 pending。",
                  "", "[详细报告](REPORT.md) · [Gallery](gallery/index.html) · [指标表](metrics.csv) · [配对统计](metrics_summary.json) · [现场状态](LIVE.json) · [清理审计](research/output_cleanup.json)"])
    temporary = R / "PROGRESS_LATEST.writing"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(R / "PROGRESS_LATEST.md")
    data = {"updated_at": stamp, "target_tracks": 8, "target_final_versions": 32, "allowed_physical_gpus": list(range(8)),
            "matrix": counts, "live_jobs": jobs, "technically_audited_complete_tracks": accepted, "production_approved": False}
    temporary = R / "research/progress_latest.writing"
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(R / "research/progress_latest.json")
