"""Finish existing conditions after separating empty-mask NA from crop loss.

Does not change model outputs, crop, camera, skeleton, weights or input video.
Keeps original failing stage state and logs intact.
"""
import asyncio
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from scan_native import ROOT, write, sha

sys.path.insert(0, str(ROOT.parent.parent / "repo"))
from fdanyone.skeleton.pipeline import Conditioning


def assess(key, job):
    source = Path(job["source"])
    package = source.parent
    root = package / "conditions"
    if (package / "READY.json").exists():
        return None
    log = ROOT / "logs" / f"{key}_conditioning.log"
    if "np.float64(0.0)" not in log.read_text():
        return None
    condition = Conditioning.load(root / "preflight/conditioning")
    meta = json.loads((condition.root / "metadata.json").read_text())
    cameras = json.loads((condition.root / "cameras.json").read_text())
    points = np.load(condition.root / "keypoints_3d.npy")
    assert np.isfinite(points).all()
    coverage = []
    for camera in cameras["cameras"]:
        w = np.asarray(camera["world_to_camera"])
        p = points @ w[:3, :3].T + w[:3, 3]
        q = p @ np.asarray(camera["K"]).T
        uv = q[..., :2] / q[..., 2:]
        good = ((p[..., 2] > 0) & (uv[..., 0] >= .06 * camera["image_width"]) &
                (uv[..., 0] <= .94 * camera["image_width"]) &
                (uv[..., 1] >= .06 * camera["image_height"]) &
                (uv[..., 1] <= .94 * camera["image_height"]))
        if not good.all():
            return None
        coverage.append({"view": camera["camera_id"], "all_joints_all_frames_inside": True})
    masks = np.load(root / "preflight/source_masks.npz")["masks"] > 12.75
    crop = meta["source_crop"]
    counts = masks.sum((1, 2))
    selected = masks[:, crop["top"]:crop["top"]+crop["height"],
                     crop["left"]:crop["left"]+crop["width"]].sum((1, 2))
    valid = counts > 0
    if not valid.any():
        return None
    retention = selected[valid] / counts[valid]
    if retention.min() < .95:
        return None
    report = {"state": "conditions_prepared_missing_mask_evidence_explicit",
              "created_at": time.time(), "original_failed_log": str(log),
              "original_failed_log_sha256": sha(log), "script_sha256": sha(Path(__file__)),
              "changed_model_outputs": False, "coverage": coverage,
              "source_mask_valid_frames": int(valid.sum()),
              "source_mask_unknown_frames": np.flatnonzero(~valid).tolist(),
              "source_mask_retention_min_on_valid_frames": float(retention.min()),
              "source_mask_retention_per_frame": [
                  float(selected[i]/counts[i]) if valid[i] else None for i in range(len(valid))],
              "quality_accepted": False, "generation_allowed": False,
              "scope": "Empty mask is unknown evidence, never counted as passing coverage. Input/skeleton remains eligible for visual and temporal review; no generated completion claim."}
    write(root / "research/empty_mask_retention.json", report)
    write(root / "preflight/review.json", {"allow_generation": False, "conditioning_prepared": True,
          "source_visibility_review": "pending", "evidence": "../research/empty_mask_retention.json"})
    return source


async def main():
    jobs = {}
    for filename in ["native_prepare_status.json", "predicted_prepare_status.json", "crop_retry_prepare_status.json", "foreground_retry_prepare_status.json", "alternative_retry_prepare_status.json", "native_mask_retry_prepare_status.json", "terminal_retry_prepare_status.json"]:
        path = ROOT / filename
        if path.exists():
            jobs.update(json.loads(path.read_text())["jobs"])
    status_path = ROOT / "retention_repair_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    semaphore = asyncio.Semaphore(2)
    async def one(key, job):
        async with semaphore:
            source = await asyncio.to_thread(assess, key, job)
            if source is None:
                return
            candidate = json.loads((source.parent / "candidate.json").read_text())
            runtime = ROOT.parent / "round2/moving61_runtime" if candidate["output_slots"] == 61 else ROOT.parent.parent / "repo"
            env = dict(os.environ, FDANYONE_SAMPLE_ROOT=str(source),
                       FDANYONE_RUNTIME=str(runtime), FDANYONE_FRAMES=str(candidate["output_slots"]),
                       OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
            with (ROOT / "logs" / f"{key}_finish_retention.log").open("x") as log:
                child = await asyncio.create_subprocess_exec(
                    "/mnt/ojc/miniconda3/envs/4danyone/bin/python",
                    str(ROOT / "scripts/finish_input.py"), env=env, stdout=log, stderr=log)
                code = await child.wait()
            status[key] = {"state": "prepared_review_pending" if code == 0 else "failed",
                           "exit_code": code, "original_conditioning_exit_retained": True}
            write(status_path, status)
    await asyncio.gather(*(one(k, v) for k, v in jobs.items()
                          if v.get("failed_stage") == "conditioning" and k not in status))


if __name__ == "__main__":
    asyncio.run(main())
