"""Analytic rigid maps with mesh and published-regressor checks for validation."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def source_maps(prepare, motion, original, local, local_to_canonical):
    w, c = motion.smpl_params_global, motion.smpl_params_incam
    assert torch.equal(w["body_pose"], c["body_pose"])
    assert torch.equal(w["betas"], c["betas"])
    with prepare.skeleton._gvhmr_geometry_context(prepare.REPO / "third_party/GVHMR"):
        from hmr4d.utils.smplx_utils import make_smplx
        model = make_smplx("supermotion").double().eval()
        with torch.inference_mode():
            shaped = model.bm.v_template[None] + torch.einsum(
                "bl,vcl->bvc", w["betas"].double(), model.bm.shapedirs)
            pelvis = torch.einsum("v,tvc->tc", model.bm.J_regressor[0], shaped).numpy()
            vg = model(**{k: v.double() for k, v in w.items()}).vertices.numpy()
            vc = model(**{k: v.double() for k, v in c.items()}).vertices.numpy()
    rg = Rotation.from_rotvec(w["global_orient"].double().numpy()).as_matrix()
    rc = Rotation.from_rotvec(c["global_orient"].double().numpy()).as_matrix()
    r = rc @ rg.transpose(0, 2, 1)
    t = c["transl"].double().numpy() + pelvis - np.einsum(
        "tij,tj->ti", r, w["transl"].double().numpy() + pelvis)
    mesh_error = float(np.max(np.abs(np.einsum("tij,tnj->tni", r, vg) + t[:, None] - vc)))
    assert mesh_error < 1e-4, mesh_error
    transform = np.repeat(np.eye(4)[None], len(r), axis=0)
    transform[:, :3, :3], transform[:, :3, 3] = r, t
    result = transform @ np.linalg.inv(original.motion_world_to_canonical_world) @ local_to_canonical
    projected = np.einsum("tij,tkj->tki", result[:, :3, :3], local.keypoints_world) + result[:, None, :3, 3]
    _, weights, _ = prepare.skeleton._load_regressor(
        prepare.resolve_regressor(None, model_dir=prepare.REPO / "models"), torch.device("cpu"))
    sums = weights.double().sum(-1).numpy()
    residual = (sums[None, :, None] - 1) * t[:, None]
    keypoint_error = float(np.max(np.abs(projected + residual - original.keypoints_incam)))
    assert keypoint_error < 1e-4, keypoint_error
    return result, dict(
        independent_all_frame_all_vertex_SE3_max_error_m=mesh_error,
        published_regressor_nonunit_semantics_residual_m=keypoint_error,
        original_tolerance_m=1e-4, weights_unchanged=True,
        scope="Rigid equivalence of estimated global/incam SMPL mesh, not correctness of real-world motion")
