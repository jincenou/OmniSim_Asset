"""Aggregate existing checkpoints' scores without creating additional media."""
import json
import statistics

from trajectory_statistics import KEYS, paired

SPECS = (("ftgs_base5k", None, 5000, 5000),
         ("ftgs_opt15k", 5000, 5000, 5000),
         ("ftgs_opt15k", 10000, 10000, 10000),
         ("ftgs_opt15k", None, 15000, 15000),
         ("4dgs_grid25", 12000, 15000, 14999),
         ("4dgs_grid25", None, 33000, 32999),
         ("4dgs_grid61", 12000, 15000, 14999),
         ("4dgs_grid61", None, 33000, 32999))


def collect(root, selection):
    rows = []
    missing = []
    for case in selection["main"] + selection["reserve"]:
        label = case["sample_id"]
        base = root / "samples" / label
        if not (base / "generation/research/generated_validation.json").exists():
            continue
        signatures = set()
        for branch, snapshot, nominal, updates in SPECS:
            name = branch if snapshot is None else f"{branch}_snapshot{snapshot}"
            path = base / "evaluations" / name / "results.json"
            if not path.exists():
                missing.append({"sample_id": label, "branch": branch, "nominal_iterations": nominal})
                continue
            result = json.loads(path.read_text())
            assert result["nominal_total_iterations"] == nominal and len(result["samples"]) == 39
            asset = json.loads((path.parent / "asset_manifest.json").read_text())
            signatures.add((result["protocol"]["all_input_pairs_sha256"], json.dumps(result["crops"], sort_keys=True)))
            rows.append({"sample_id": label, "cohort": case["cohort"], "recipe": branch,
                         "branch": f"{branch}@{nominal}", "snapshot": snapshot,
                         "nominal_iterations": nominal, "actual_updates_represented": updates,
                         "gaussians": asset["point_count"], "result": str(path), **result["mean"]})
        assert len(signatures) <= 1, f"Budget targets/crops differ for {label}"
    cohorts = {}
    for cohort in ("main", "reserve"):
        selected = [row for row in rows if row["cohort"] == cohort]
        groups = {}
        for token in sorted({row["branch"] for row in selected}):
            values = [row for row in selected if row["branch"] == token]
            groups[token] = {"tracks": len(values), "sample_ids": [row["sample_id"] for row in values],
                             "actual_updates_represented": values[0]["actual_updates_represented"],
                             "mean": {key: statistics.mean(row[key] for row in values) for key in KEYS},
                             "median": {key: statistics.median(row[key] for row in values) for key in KEYS}}
        comparisons = (("ftgs_opt15k@5000", "ftgs_opt15k@15000"),
                       ("ftgs_opt15k@10000", "ftgs_opt15k@15000"),
                       ("4dgs_grid25@15000", "4dgs_grid25@33000"),
                       ("4dgs_grid61@15000", "4dgs_grid61@33000"))
        cohorts[cohort] = {"milestones": groups, "within_same_run_changes": {
            right + "_minus_" + left: paired(selected, left, right) for left, right in comparisons}}
    return {"scope": "Existing generated-heldout checkpoint evaluations; trajectory first; no checkpoint selected using heldout scores.",
            "budget_caution": "HUST nominal15k nativepointcloud is pre-step14999 updates; final33k is32999. FTGS snapshots are post-step. Same update count is not matched GPU time or capacity.",
            "recipe_caution": "FTGS original5k is a separate cold run and full5k LR plan; FTGS-opt15k's5k snapshot cannot replace it. Within-run changes include the fixed LR schedule, relocation/densification and natural capacity changes.",
            "time_caution": "Full-run elapsed times in resource_table are not intermediate-snapshot runtime measurements.",
            "expected_rows_for_eight_tracks": 64, "available_rows": len(rows), "missing": missing,
            "rows": rows, "cohorts": cohorts}


def markdown(summary):
    budget = summary["budget_comparison"]
    lines = ["", "### 训练预算快照", "", f"已汇总 {budget['available_rows']}/64 行终态及中间预算指标；只复用现有数值，没有新增媒体。",
             "", "|组别|配方 / 名义迭代|实际更新|已有轨迹数|PSNR均值↑|LPIPS均值↓|IoU均值↑|",
             "|---|---|---:|---:|---:|---:|---:|"]
    for cohort, title in (("main", "主样本"), ("reserve", "备用补充")):
        for token, row in budget["cohorts"][cohort]["milestones"].items():
            mean = row["mean"]
            lines.append(f"|{title}|{token}|{row['actual_updates_represented']}|{row['tracks']}|{mean['full_psnr']:.5f}|{mean['crop_lpips']:.5f}|{mean['silhouette_iou']:.5f}|")
    lines.extend(["", "各行可用人数可能不同，不能直接相减作配对结论。同一人的5k→15k、10k→15k、总15k→33k变化及改善/退化人数已另存于 metrics_summary.json 的 budget_comparison.cohorts。HUST总15k原生快照代表14999次更新，FTGS15k代表15000次；同更新次数也不等于同GPU耗时。FTGS原5k与优化15k中的5k快照使用不同学习率计划和初始时间宽度。", ""])
    return lines
