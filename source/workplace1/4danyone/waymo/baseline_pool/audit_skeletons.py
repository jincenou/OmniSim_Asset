"""Fast temporal diagnostics; flags prioritize review, never reject occlusion."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time

import numpy as np

from scan_native import ROOT, write, sha
from package_catalog import ready_paths

POLICY = {
    "reprojection_median_over_box_height_review": .12,
    "reprojection_p95_over_box_height_review": .30,
    "median_confident_coco_points_review_below": 8,
    "local_joint_step_p99_over_body_height_review": .12,
    "knee_bend_step_max_deg_review": 45,
    "knee_bend_max_deg_context_review": 120,
    "scope": "Engineering review priorities, not calibrated acceptance thresholds. True fast actions/crouching remain eligible.",
    "occlusion_exclusion": False,
}


def audit(package):
    ready = json.loads((package / "READY.json").read_text())
    source = package / "clip0"
    condition = package / ready["conditioning"]
    meta = json.loads((condition / "metadata.json").read_text())
    names = meta["keypoint_names"]
    points = np.load(condition / "keypoints_3d.npy")
    assert points.shape[0] == meta["num_frames"] and points.shape[1] == len(names)
    assert np.isfinite(points).all()
    hip = .5 * (points[:, names.index("left-hip")] + points[:, names.index("right-hip")])
    local = points - hip[:, None]
    height = float(np.median(np.ptp(points[..., 1], axis=1)))
    assert height > 0
    major = [names.index(n) for n in ["left-hip", "right-hip", "left-knee", "right-knee",
                                      "left-ankle", "right-ankle", "left-shoulder", "right-shoulder",
                                      "left-elbow", "right-elbow", "left-wrist", "right-wrist"]]
    step = np.linalg.norm(np.diff(local[:, major], axis=0), axis=-1) / height
    bends = {}
    for side in ["left", "right"]:
        h, k, a = (points[:, names.index(f"{side}-{joint}")] for joint in ["hip", "knee", "ankle"])
        first, second = h-k, a-k
        cosine = (first*second).sum(-1) / np.maximum(np.linalg.norm(first, axis=-1)*np.linalg.norm(second, axis=-1), 1e-12)
        bends[side] = 180 - np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
    metrics = json.loads((source / "motion_pair/comparison.json").read_text())["metrics"]["calibrated_moving"]
    flags = []
    for key, threshold in [
        ("median_detector_reprojection_over_bbox_height", POLICY["reprojection_median_over_box_height_review"]),
        ("p95_detector_reprojection_over_bbox_height", POLICY["reprojection_p95_over_box_height_review"])]:
        if metrics[key] > threshold:
            flags.append("large_" + key)
    if metrics["confident_points_per_frame_min_median"][1] < POLICY["median_confident_coco_points_review_below"]:
        flags.append("sparse_observed_keypoint_support")
    p99 = float(np.quantile(step, .99))
    if p99 > POLICY["local_joint_step_p99_over_body_height_review"]:
        flags.append("fast_local_joint_change_check_against_real_action")
    max_step = max(float(np.abs(np.diff(v)).max()) for v in bends.values())
    if max_step > POLICY["knee_bend_step_max_deg_review"]:
        flags.append("abrupt_knee_bend_change")
    if max(float(v.max()) for v in bends.values()) > POLICY["knee_bend_max_deg_context_review"]:
        flags.append("deep_knee_bend_check_crouch_or_estimation_error")
    masks = json.loads((source / "research/batch_gate_mask.json").read_text())
    report = {"created_at": time.time(), "policy": POLICY, "script_sha256": sha(Path(__file__)),
              "state": "review_priority" if flags else "no_numeric_flag_visual_review_pending",
              "flags": flags, "automatic_rejection": False, "quality_accepted": False,
              "occlusion_exclusion": False, "source_mask_risk_flags": masks.get("risk_flags", masks.get("reasons", [])),
              "local_joint_step_p99_over_median_body_height": p99,
              "max_knee_bend_step_deg": max_step,
              "knee_bend_range_deg": {k: [float(v.min()), float(v.max())] for k, v in bends.items()},
              "reprojection": metrics,
              "limitations": ["Regressed 3D joints and 2D detector consistency are not pose ground truth.",
                             "Low errors or no flag do not prove hidden limbs correct.",
                             "Real crouching/running must not be excluded solely by these flags."]}
    write(package / "skeleton_audit.json", report)
    return report


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch-pid", nargs="*", type=int, default=[])
    args = parser.parse_args()
    write(ROOT / "SKELETON_REVIEW_POLICY.json", POLICY)
    while True:
        counts, flagged, failures = {}, [], {}
        for ready in ready_paths():
            target = ready.parent / "skeleton_audit.json"
            try:
                report = json.loads(target.read_text()) if target.exists() else audit(ready.parent)
                counts[report["state"]] = counts.get(report["state"], 0) + 1
                if report["flags"]:
                    flagged.append({"package": str(ready.parent), "flags": report["flags"]})
            except Exception as exc:
                failures[str(ready.parent)] = repr(exc)
        write(ROOT / "skeleton_review_summary.json", {
            "counts": counts, "flagged_packages": flagged, "diagnostic_failures": failures,
            "quality_passed_total": None, "occlusion_alone_rejected": 0})
        live = False
        for pid in args.watch_pid:
            try:
                os.kill(pid, 0)
                command = Path(f"/proc/{pid}/cmdline").read_bytes()
                if b"run_native_pool.py" in command or b"run_predicted_pool.py" in command:
                    live = True
            except (ProcessLookupError, FileNotFoundError):
                pass
        if not live:
            break
        await asyncio.sleep(45)


if __name__ == "__main__":
    asyncio.run(main())
