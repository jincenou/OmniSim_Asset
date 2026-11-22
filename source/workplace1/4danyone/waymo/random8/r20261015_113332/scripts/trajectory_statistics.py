"""Descriptive, per-trajectory statistics with explicit matched denominators."""
import statistics

BRANCHES = ("ftgs_base5k", "ftgs_opt15k", "4dgs_grid25", "4dgs_grid61")
KEYS = ("full_psnr", "ssim", "foreground_psnr", "crop_lpips", "silhouette_iou", "outside_alpha_mean")


def branch_statistics(rows):
    result = {}
    for branch in BRANCHES:
        selected = [row for row in rows if row["branch"] == branch]
        result[branch] = {"available_tracks": len(selected), "sample_ids": [row["sample_id"] for row in selected],
                          "mean": {key: statistics.mean(row[key] for row in selected) if selected else None for key in KEYS},
                          "median": {key: statistics.median(row[key] for row in selected) if selected else None for key in KEYS}}
    return result


def paired(rows, left, right):
    indexed = {(row["sample_id"], row["branch"]): row for row in rows}
    ids = sorted({row["sample_id"] for row in rows if (row["sample_id"], left) in indexed and (row["sample_id"], right) in indexed})
    result = {"left": left, "right": right, "delta_definition": "right_minus_left", "paired_tracks": len(ids), "sample_ids": ids, "metrics": {}}
    for key in KEYS:
        changes = {label: indexed[label, right][key] - indexed[label, left][key] for label in ids}
        sign = -1 if key in ("crop_lpips", "outside_alpha_mean") else 1
        result["metrics"][key] = {"per_track_change": changes,
                                 "mean_change": statistics.mean(changes.values()) if changes else None,
                                 "median_change": statistics.median(changes.values()) if changes else None,
                                 "improved": sum(value * sign > 0 for value in changes.values()),
                                 "degraded": sum(value * sign < 0 for value in changes.values()),
                                 "unchanged": sum(value == 0 for value in changes.values())}
    return result


def grouped_statistics(rows):
    groups = {}
    for field in ("human_action", "ego_group", "frames"):
        groups[field] = {}
        for value in sorted({row[field] for row in rows}, key=str):
            subset = [row for row in rows if row[field] == value]
            groups[field][str(value)] = {"tracks": len({row["sample_id"] for row in subset}),
                                        "branches": branch_statistics(subset),
                                        "paired_to_ftgs_base5k": {branch: paired(subset, BRANCHES[0], branch) for branch in BRANCHES[1:]},
                                        "grid61_minus_grid25": paired(subset, BRANCHES[2], BRANCHES[3]),
                                        "rows": subset}
    return groups


def extend_summary(root, selection, rows, stats):
    denominators = {}
    for cohort in ("main", "reserve"):
        cases = selection[cohort]
        labels = [case["sample_id"] for case in cases]
        attempted = [label for label in labels if (root / "samples" / label / "source/native.json").exists()]
        generated = [label for label in attempted if (root / "samples" / label / "generation/research/generated_validation.json").exists()]
        denominators[cohort] = {"preregistered": len(cases), "observed_attempted": len(attempted),
                                "attempted_ids": attempted, "legal_24view_ids": generated,
                                "attempted_without_legal_24view_ids": [label for label in attempted if label not in generated],
                                "unused_ids": [label for label in labels if label not in attempted]}
        subset = [row for row in rows if row["cohort"] == cohort]
        stats[cohort]["groups"] = grouped_statistics(subset)
        stats[cohort]["paired_to_ftgs_base5k"] = {branch: paired(subset, BRANCHES[0], branch) for branch in BRANCHES[1:]}
        stats[cohort]["grid61_minus_grid25"] = paired(subset, BRANCHES[2], BRANCHES[3])
        indexed = {(row["sample_id"], row["branch"]): row for row in subset}
        complete = [label for label in labels if all((label, branch) in indexed for branch in BRANCHES)]
        stats[cohort]["matched_complete_four"] = {"tracks": len(complete), "sample_ids": complete,
            "branches": branch_statistics([row for row in subset if row["sample_id"] in complete])}
    return denominators


def markdown(summary):
    lines = ["", "### 同一批完整四组人物的配对摘要", "",
             "以下各方法共享相同人物分母；主样本与备用补充样本分开。动作分组是对实拍的回顾性描述，不是官方行为标签。", "",
             "|组别 / 完整配对人数|方法|PSNR均值 / 中位数↑|LPIPS均值 / 中位数↓|IoU均值 / 中位数↑|",
             "|---|---|---:|---:|---:|"]
    for cohort, label in (("main", "主样本"), ("reserve", "备用补充")):
        matched = summary["cohorts"][cohort]["matched_complete_four"]
        for branch in BRANCHES:
            row = matched["branches"][branch]
            values = [f"{row['mean'][key]:.5f} / {row['median'][key]:.5f}" if row["available_tracks"] else "—" for key in ("full_psnr", "crop_lpips", "silhouette_iou")]
            lines.append(f"|{label} n={matched['tracks']}|{branch}|" + "|".join(values) + "|")
    lines.extend(["", "六项指标的均值、中位数、逐人配对变化、改善/退化人数及动作/自车运动/片长分组详见 [metrics_summary.json](metrics_summary.json)。样本数随实际完成进度更新；当前结果不能外推到整个 Waymo。", ""])
    return lines
