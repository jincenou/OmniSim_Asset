"""CPU-only numerical diagnosis on saved motion; no production artifact changes."""
import json
import functools
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1] / "repo"
sys.path[:0] = [str(REPO), str(ROOT / "scripts")]
from fdanyone.assets import resolve_regressor
from fdanyone.motion.result import MotionResult
from fdanyone.skeleton import pipeline
from geometry_precision import build
from root_local_geometry import separate_horizontal_root
from scan_native import sha, write


def rigid_error(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    ac, bc = a.mean(1, keepdims=True), b.mean(1, keepdims=True)
    u, _, vt = np.linalg.svd(np.swapaxes(a-ac, 1, 2) @ (b-bc))
    fix = np.repeat(np.eye(3)[None], len(a), axis=0)
    fix[:, 2, 2] = np.linalg.det(np.swapaxes(vt, 1, 2) @ np.swapaxes(u, 1, 2))
    rotation = np.swapaxes(vt, 1, 2) @ fix @ np.swapaxes(u, 1, 2)
    translation = bc[:, 0]-np.einsum("tij,tj->ti", rotation, ac[:, 0])
    residual = np.einsum("tij,tkj->tki", rotation, a) + translation[:, None] - b
    return {"max_abs": float(np.abs(residual).max()),
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "frames_above_original_1e4_limit": int((np.abs(residual).max((1, 2)) >= 1e-4).sum())}


def main():
    torch.set_num_threads(2)
    regressor = resolve_regressor(None, model_dir=REPO / "models")
    weights = torch.load(regressor, map_location="cpu", weights_only=True)["weights"]
    results = []
    validate = MotionResult.validate
    for queue in ["native_prepare_status.json", "predicted_prepare_status.json"]:
        for key, job in json.loads((ROOT/queue).read_text())["jobs"].items():
            if job.get("failed_stage") != "conditioning":
                continue
            log = (ROOT/"logs"/f"{key}_conditioning.log").read_text()
            if "Global/incam SMPL geometry is not a consistent rigid pair" not in log:
                continue
            source = Path(job["source"])
            count = json.loads((source.parent/"candidate.json").read_text())["output_slots"]
            assert count in (61, 121)
            # The existing61 runtime differs here only in this default; keep
            # the original validation body and enforce the manifest's count.
            MotionResult.validate = functools.partialmethod(validate, expected_frames=count)
            motion_path = source/"motion_pair/calibrated_moving"
            before = sha(motion_path/"motion.safetensors")
            motion = MotionResult.load(motion_path)
            stable = build(pipeline._body_geometry)
            original_loader = stable.__globals__["_load_regressor"]
            raw = stable(motion, regressor, REPO/"third_party/GVHMR", "cpu")
            raw_local, _, _ = separate_horizontal_root(raw)
            def normalized_loader(path, device):
                support, w, metadata = original_loader(path, device)
                w = w.double()
                return support, w/w.sum(1, keepdim=True), metadata
            stable.__globals__["_load_regressor"] = normalized_loader
            normalized = stable(motion, regressor, REPO/"third_party/GVHMR", "cpu")
            normalized_local, _, _ = separate_horizontal_root(normalized)
            assert sha(motion_path/"motion.safetensors") == before
            result = {
                "review_id": key, "motion_sha256": before,
                "body_pose_max_difference": float((motion.smpl_params_global["body_pose"]-motion.smpl_params_incam["body_pose"]).abs().max()),
                "betas_max_difference": float((motion.smpl_params_global["betas"]-motion.smpl_params_incam["betas"]).abs().max()),
                "maximum_incam_translation_norm": float(motion.smpl_params_incam["transl"].norm(dim=-1).max()),
                "raw_keypoint_rigid_fit": rigid_error(raw_local.keypoints_world, raw.keypoints_incam),
                "normalized_weight_diagnostic_fit": rigid_error(normalized_local.keypoints_world, normalized.keypoints_incam),
                "max_keypoint_change_world": float(np.abs(raw.keypoints_world-normalized.keypoints_world).max()),
                "max_keypoint_change_incam": float(np.abs(raw.keypoints_incam-normalized.keypoints_incam).max()),
                "production_artifacts_changed": False,
            }
            results.append(result)
            write(ROOT/"research/rigid_numeric_diagnosis.json", {
                "state": "running", "results": results, "script_sha256": sha(Path(__file__)),
                "regressor_sha256": sha(regressor),
                "max_regressor_row_sum_error": float((weights.double().sum(1)-1).abs().max()),
                "scope": "CPU diagnostic normalization only, not an admitted baseline change"})
            print(json.dumps(result), flush=True)
    report = json.loads((ROOT/"research/rigid_numeric_diagnosis.json").read_text())
    report.update(state="finished_requires_interpretation", finished=time.time())
    write(ROOT/"research/rigid_numeric_diagnosis.json", report)


if __name__ == "__main__":
    main()
