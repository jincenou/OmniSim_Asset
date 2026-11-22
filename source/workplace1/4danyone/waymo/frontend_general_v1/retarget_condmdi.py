"""Fit local CondMDI proposals to the existing person's SMPL-X skeleton.

Model yaw/trajectory are not adopted. Visible original rotations remain exact
at this proposal stage; a separate fixed O2 stage performs source-camera fitting.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import smplx
from smplx.lbs import blend_shapes, vertices2joints, batch_rodrigues, batch_rigid_transform
from scipy.spatial.transform import Rotation, Slerp
from safetensors.torch import load_file, save_file
from condmdi_mask import PARENTS
from condmdi_paths import frozen_protocol, case_paths

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor/CondMDI"))
from data_loaders.humanml.scripts.motion_process import process_file, recover_root_rot_pos
from optimize_motion import supported_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("branch", choices=["no_text", "text"])
    parser.add_argument("--suite", choices=["counterfactual", "real"], default="counterfactual")
    args = parser.parse_args()
    torch.set_num_threads(2)
    protocol_path, cfg = frozen_protocol(args.suite, "RETARGET_PROTOCOL")
    assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == protocol_path.with_suffix(".sha256").read_text().strip()
    assert args.case in cfg["cases"]
    assert torch.cuda.device_count() == 1
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    assert uuid == os.environ["FD_EXPECTED_GPU_UUID"].removeprefix("GPU-")
    source, case = case_paths(args.suite, args.case)
    out = case / args.branch
    assert not (out / "retargeted.safetensors").exists()
    data = np.load(case / "condition.npz")
    proposal = np.load(out / "proposal.npz")
    mapping = json.loads((case / "retarget_geometry.json").read_text())
    state = load_file(str(source / "motion/motion.safetensors"))
    tensor = lambda x: torch.tensor(x, dtype=torch.float32, device="cuda")
    original = state["smpl_params_global.body_pose"].numpy()
    np.testing.assert_array_equal(original, state["smpl_params_incam.body_pose"].numpy())
    n = len(original)
    t25 = np.arange(n) / 25.
    t20 = np.arange(round(t25[-1] * 20) + 1) / 20.
    raw = {k: state["smpl_params_global." + k].numpy() for k in ["body_pose", "global_orient", "transl", "betas"]}
    pose20 = np.stack([Slerp(t25, Rotation.from_rotvec(original.reshape(n, 21, 3)[:, j]))(t20).as_rotvec()
                       for j in range(21)], axis=1).reshape(len(t20), 63)
    root20 = Slerp(t25, Rotation.from_rotvec(raw["global_orient"]))(t20).as_rotvec()
    trans20 = np.stack([np.interp(t20, t25, raw["transl"][:, j]) for j in range(3)], axis=1)
    betas = raw["betas"].mean(0, keepdims=True)
    body = smplx.create(str(ROOT.parents[1] / "repo/models/body_models"), model_type="smplx",
                        gender="neutral", flat_hand_mean=True, use_pca=False,
                        batch_size=len(t20)).eval().cuda()
    body.requires_grad_(False)
    started = time.time()
    with torch.inference_mode():
        joints20 = body(body_pose=tensor(pose20), global_orient=tensor(root20), transl=tensor(trans20),
                        betas=tensor(np.repeat(betas, len(t20), axis=0))).joints[:, :22].cpu()
    os.chdir(ROOT / "vendor/CondMDI")
    features, canonical, _, _ = process_file(joints20, .002)
    np.testing.assert_allclose(canonical[:-1], data["estimated_canonical_joints"], atol=2e-5, rtol=0)
    _, _, original_yaw = recover_root_rot_pos(torch.tensor(features, dtype=torch.float32), return_rot_ang=True)
    predicted = proposal["joints"]
    delta_yaw = 2 * (proposal["features"][:, 0] - original_yaw.numpy())
    # HumanML root quaternion maps world to local; replace only this proposal's
    # yaw with the original estimated yaw before inverse canonical rotation.
    yaw_rotation = Rotation.from_euler("y", delta_yaw[:, None]).as_matrix()
    bones = predicted[:, 1:] - predicted[:, PARENTS[1:]]
    bones = np.einsum("tij,tkj->tki", yaw_rotation, bones)
    row_rotation = np.array(mapping["source_to_standard_row_rotation"])
    bones = bones @ row_rotation.T
    norms = np.linalg.norm(bones, axis=-1, keepdims=True)
    assert norms.min() > 1e-7
    directions20 = bones / norms
    times = data["time_20fps"]
    available = t25 <= times[-1] + 1e-8
    flat = directions20.reshape(len(times), -1)
    targets = np.stack([np.interp(t25[available], times, flat[:, k]) for k in range(flat.shape[1])], axis=1).reshape(-1, 21, 3)
    targets /= np.maximum(np.linalg.norm(targets, axis=-1, keepdims=True), 1e-8)
    visibility_cfg = json.loads((ROOT / "protocol_v2.json").read_text())
    visible, _ = supported_points(state["observed_keypoints_2d"].numpy(),
                                  np.load(source / "input/target_masks.npz")["masks"],
                                  visibility_cfg["visibility"])
    changed = np.zeros((n, 21), dtype=bool)
    for joint, child in visibility_cfg["O1"]["changed_joint_indices_body_pose_to_coco_child"].items():
        changed[:, int(joint)] = ~visible[:, child] & available
    with torch.no_grad():
        shaped = body.v_template + blend_shapes(tensor(betas), body.shapedirs)
        rest = vertices2joints(body.J_regressor, shaped)[:, :22].expand(n, -1, -1).contiguous()
    source_root = tensor(raw["global_orient"])
    source_trans = tensor(raw["transl"])
    pose0 = tensor(original.reshape(n, 21, 3))
    mask = tensor(changed[..., None])
    delta = torch.nn.Parameter(torch.zeros_like(pose0))
    parents = body.parents[:22]

    def geometry(pose):
        rot = batch_rodrigues(torch.cat([source_root[:, None], pose], dim=1).reshape(-1, 3)).reshape(n, 22, 3, 3)
        joints, _ = batch_rigid_transform(rot, rest, parents)
        joints = joints + source_trans[:, None]
        b = joints[:, 1:] - joints[:, PARENTS[1:]]
        return joints, torch.nn.functional.normalize(b, dim=-1), rot[:, 1:]

    with torch.no_grad():
        initial_joints, initial_directions, initial_rot = geometry(pose0)
        # Check the efficient 22-joint FK against the actual SMPL-X body layer.
        take = np.array([0, n // 2, n - 1])
        full = smplx.create(str(ROOT.parents[1] / "repo/models/body_models"), model_type="smplx",
                            gender="neutral", flat_hand_mean=True, use_pca=False, batch_size=3).eval().cuda()
        actual = full(body_pose=tensor(original[take]), global_orient=source_root[take],
                      transl=source_trans[take], betas=tensor(np.repeat(betas, 3, axis=0))).joints[:, :22]
        fk_error = float((actual - initial_joints[take]).abs().max())
        assert fk_error < 1e-5
    target = initial_directions.detach().clone()
    target[available] = tensor(targets)
    # Only bones downstream of permitted variable rotations can be fitted.
    active = np.zeros((n, 21), dtype=bool)
    for joint in range(1, 22):
        ancestor = int(PARENTS[joint])
        while ancestor > 0:
            active[:, joint - 1] |= changed[:, ancestor - 1]
            ancestor = int(PARENTS[ancestor])
    weights = tensor(active)
    fit_steps = cfg["steps"] if weights.sum() > 0 else 0
    optimizer = torch.optim.Adam([delta], lr=cfg["learning_rate"])
    trace = []
    for step in range(fit_steps):
        optimizer.zero_grad(set_to_none=True)
        pose = pose0 + delta * mask
        _, direction, rot = geometry(pose)
        fit = ((direction - target).square().sum(-1) * weights).sum() / weights.sum()
        anchor = (((rot - initial_rot).square().sum((-1, -2)) / 2) * mask[..., 0]).sum() / mask.sum().clamp_min(1)
        loss = fit + cfg["rotation_anchor_weight"] * anchor
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        if step in [0, cfg["steps"] - 1]:
            trace.append(dict(step=step + 1, fit=float(fit.detach()), anchor=float(anchor.detach())))
    fitted = (pose0 + delta * mask).detach().cpu().numpy()
    np.testing.assert_array_equal(fitted[~changed], original.reshape(n, 21, 3)[~changed])
    save_file({"gap_proposal_25fps": torch.tensor(fitted.reshape(n, 63)),
               "changed_joint_mask": torch.tensor(changed),
               "proposal_available": torch.tensor(available)},
              str(out / "retargeted.safetensors"))
    report = dict(state="subject_pose_proposal_created_needs_source_alignment", case=args.case,
                  branch=args.branch, suite=args.suite, gpu_uuid=uuid, steps=fit_steps, trace=trace,
                  full_smplx_vs_fk_max_error_m=fk_error, changed_joint_frames=int(changed.sum()),
                  unchanged_rotations_bitwise_preserved=True, root_and_shape_unchanged=True,
                  generated_root_yaw_replaced_by_original_estimated_root_yaw=True,
                  unavailable_slots=np.flatnonzero(~available).tolist(), quality_approved=False,
                  elapsed_seconds=time.time() - started,
                  scope="Local pose retargeting only; no true hidden-motion or world-trajectory claim",
                  protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest())
    (out / "retarget_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
