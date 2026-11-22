"""Use analytic SMPL root transforms; verify non-unit regressor semantics explicitly.

The original preparation code, model parameters, keypoint weights, rendering, and
acceptance tolerances remain unchanged. This fixes the sidecar consistency check.
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import prepare_dev6_generation as prepare

LAST = {}
CHECKS = []
original_build = prepare.build


def build(fn):
    original = original_build(fn)
    def wrapped(motion, *args, **kwargs):
        LAST["motion"] = motion
        return original(motion, *args, **kwargs)
    return wrapped


def maps(original, local, local_to_canonical):
    motion = LAST["motion"]
    w, c = motion.smpl_params_global, motion.smpl_params_incam
    assert torch.equal(w["body_pose"], c["body_pose"])
    assert torch.equal(w["betas"], c["betas"])
    # Only rest pelvis is evaluated here; full mesh equivalence has an independent audit.
    with prepare.skeleton._gvhmr_geometry_context(prepare.REPO / "third_party/GVHMR"):
        from hmr4d.utils.smplx_utils import make_smplx
        model = make_smplx("supermotion").double().eval()
        with torch.no_grad():
            shaped = model.bm.v_template[None] + torch.einsum(
                "bl,vcl->bvc", w["betas"].double(), model.bm.shapedirs)
            pelvis = torch.einsum("v,tvc->tc", model.bm.J_regressor[0], shaped).numpy()
    rg = Rotation.from_rotvec(w["global_orient"].double().numpy()).as_matrix()
    rc = Rotation.from_rotvec(c["global_orient"].double().numpy()).as_matrix()
    r = rc @ np.swapaxes(rg, 1, 2)
    t = c["transl"].double().numpy() + pelvis - np.einsum(
        "tij,tj->ti", r, w["transl"].double().numpy() + pelvis)
    transform = np.repeat(np.eye(4)[None], len(r), axis=0)
    transform[:, :3, :3], transform[:, :3, 3] = r, t
    result = transform @ np.linalg.inv(original.motion_world_to_canonical_world) @ local_to_canonical
    projected = np.einsum("tij,tkj->tki", result[:, :3, :3], local.keypoints_world) + result[:, None, :3, 3]
    support, weights, _ = prepare.skeleton._load_regressor(
        prepare.resolve_regressor(None, model_dir=prepare.REPO / "models"), torch.device("cpu"))
    sums = weights.double().sum(-1).numpy()
    weighted_residual = (sums[None, :, None] - 1) * t[:, None]
    error = float(np.abs(projected + weighted_residual - original.keypoints_incam).max())
    assert error < 1e-4, f"Analytic mesh map with published weight sums fails unchanged tolerance: {error}"
    CHECKS.append(dict(
        map_method="Analytic SMPL root SE3 with rest pelvis, independent of regressed keypoint Kabsch",
        ordinary_keypoint_residual_max_abs_m=float(np.abs(projected - original.keypoints_incam).max()),
        nonunit_weight_residual_accounted_max_abs_m=error,
        original_check_tolerance_m=1e-4, published_regressor_weights_unchanged=True,
        rigid_map_is_for_mesh=True,
        caution="Published non-unit weighted keypoints are not exact affine points; do not call their ordinary residual a mesh inconsistency"))
    return result, error


if __name__ == "__main__":
    sample = sys.argv[1]
    diagnosis_path = prepare.B / "rigid_regressor_diagnosis.json"
    diagnosis = json.loads(diagnosis_path.read_text())
    assert diagnosis["sample"] == sample
    for branch in diagnosis["branches"].values():
        assert branch["torch.float32"]["mesh_root_SE3_max_abs_m"] < 1e-4
        assert branch["torch.float64"]["residual_after_nonunit_weight_accounting_max_abs_m"] < 1e-4
    prepare.build = build
    prepare.source_camera_maps = maps
    prepare.main()
    report_path = prepare.B / sample / "pairing.json"
    report = json.loads(report_path.read_text())
    assert len(CHECKS) == 2
    for label, check in zip(("G1", "G2"), CHECKS, strict=True):
        report["reports"][label].pop("source_rigid_error")
        report["reports"][label]["source_geometry_check"] = check
        report["reports"][label]["independent_mesh_SE3_max_abs_m"] = diagnosis["branches"][label]["torch.float32"]["mesh_root_SE3_max_abs_m"]
    report["geometry_diagnosis"] = str(diagnosis_path)
    prepare.save(report_path, report)
