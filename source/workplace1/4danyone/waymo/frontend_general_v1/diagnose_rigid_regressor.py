"""Check root SE3 on true mesh versus non-unit published vertex-regressor sums."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from safetensors.torch import load_file

G = Path(__file__).resolve().parent
REPO = G.parent.parent / "repo"
sys.path.insert(0, str(REPO))
from fdanyone.skeleton import pipeline as skeleton
from fdanyone.assets import resolve_regressor


def root_map(world, camera, pelvis):
    rg = Rotation.from_rotvec(world["global_orient"].double().numpy()).as_matrix()
    rc = Rotation.from_rotvec(camera["global_orient"].double().numpy()).as_matrix()
    r = rc @ np.swapaxes(rg, 1, 2)
    t = camera["transl"].double().numpy() + pelvis - np.einsum(
        "tij,tj->ti", r, world["transl"].double().numpy() + pelvis)
    result = np.repeat(np.eye(4)[None], len(r), axis=0)
    result[:, :3, :3], result[:, :3, 3] = r, t
    return result


def main():
    torch.set_num_threads(4)
    p = json.loads((G / "generation_dev6/PROTOCOL.json").read_text())
    sample = "P04"
    source = Path(p["sources"][sample])
    original = load_file(str(source / "motion/motion.safetensors"))
    aligned = load_file(str(G / f"text_motion/real/cases/{sample}/text/aligned/motion_candidates.safetensors"))
    world = {k.split(".", 1)[1]: v for k, v in original.items() if k.startswith("smpl_params_global.")}
    camera = {k.split(".", 1)[1]: v for k, v in original.items() if k.startswith("smpl_params_incam.")}
    world["betas"], camera["betas"] = aligned["betas"], aligned["betas"]
    support, weights, _ = skeleton._load_regressor(resolve_regressor(None, model_dir=REPO / "models"), torch.device("cpu"))
    weights = weights.double()
    sums = weights.sum(-1).numpy()
    results = {}
    with skeleton._gvhmr_geometry_context(REPO / "third_party/GVHMR"):
        from hmr4d.utils.smplx_utils import make_smplx
        model = make_smplx("supermotion").eval()
        for label in ("G1", "G2"):
            wg = world if label == "G1" else dict(world, body_pose=aligned["body_pose"])
            ci = camera if label == "G1" else dict(camera, body_pose=aligned["body_pose"],
                global_orient=aligned["incam.global_orient"], transl=aligned["incam.transl"])
            # Actual float32 runtime and separate double evaluation of exactly the same stored coefficients.
            records = {}
            for dtype in (torch.float32, torch.float64):
                model = model.to(dtype=dtype)
                a = {k: v.to(dtype=dtype) for k, v in wg.items()}
                b = {k: v.to(dtype=dtype) for k, v in ci.items()}
                with torch.inference_mode():
                    vg = model(**a).vertices.double()
                    vc = model(**b).vertices.double()
                    # Use the model's original full template/regressor, avoiding float32 cached pelvis buffers.
                    shaped = model.bm.v_template[None] + torch.einsum("bl,vcl->bvc", a["betas"], model.bm.shapedirs)
                    pelvis = torch.einsum("v,tvc->tc", model.bm.J_regressor[0], shaped).double().numpy()
                    transform = root_map(wg, ci, pelvis)
                    rg, tr = transform[:, :3, :3], transform[:, :3, 3]
                    mesh_error = np.einsum("tij,tvj->tvi", rg, vg.numpy()) + tr[:, None] - vc.numpy()
                    kg = (vg[:, support] * weights[None, :, :, None]).sum(2).numpy()
                    kc = (vc[:, support] * weights[None, :, :, None]).sum(2).numpy()
                    ordinary = np.einsum("tij,tkj->tki", rg, kg) + tr[:, None] - kc
                    predicted = -(sums[None, :, None] - 1) * tr[:, None]
                    records[str(dtype)] = {
                        "mesh_root_SE3_max_abs_m": float(np.abs(mesh_error).max()),
                        "published_keypoints_ordinary_SE3_max_abs_m": float(np.abs(ordinary).max()),
                        "nonunit_weight_predicted_residual_max_abs_m": float(np.abs(predicted).max()),
                        "residual_after_nonunit_weight_accounting_max_abs_m": float(np.abs(ordinary - predicted).max()),
                        "incam_translation_max_abs_m": float(np.abs(ci["transl"].numpy()).max()),
                    }
            results[label] = records
    report = dict(sample=sample, scope="Read-only geometry diagnosis; no thresholds or model weights changed",
                  max_abs_weight_sum_minus_one=float(np.abs(sums - 1).max()), branches=results)
    output = G / "generation_dev6/rigid_regressor_diagnosis.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
