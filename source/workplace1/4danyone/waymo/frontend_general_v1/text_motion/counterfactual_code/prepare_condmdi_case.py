"""Prepare masked Waymo estimates for CondMDI; no hidden reference image access."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch
import smplx
from safetensors.torch import load_file
from condmdi_mask import observation_mask, normalized_condition

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "vendor/CondMDI"
sys.path.insert(0, str(REPO))
from data_loaders.humanml.scripts.motion_process import process_file, recover_root_rot_pos, recover_from_ric
from optimize_motion import supported_points

SMPL_NAMES = ("pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", "right_ankle",
              "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
              "left_elbow", "right_elbow", "left_wrist", "right_wrist")
COCO_NAMES = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
              "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
              "right_knee", "left_ankle", "right_ankle")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    args = parser.parse_args()
    protocol = json.loads((ROOT / "text_motion/PROTOCOL.json").read_text())
    assert args.case in protocol["cases"]
    source = ROOT / "occlusion_v1" / args.case / "rgb"
    sample = args.case.split("_")[0]
    out = ROOT / "text_motion/cases" / args.case
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    state = load_file(str(source / "motion/motion.safetensors"))
    keys = ("body_pose", "global_orient", "transl", "betas")
    params = {k: state["smpl_params_global." + k].numpy() for k in keys}
    n = len(params["transl"])
    source_times = np.arange(n) / 25.
    times = np.arange(round(source_times[-1] * 20) + 1) / 20.
    assert times[-1] <= source_times[-1] + 1e-8
    pose = params["body_pose"].reshape(n, 21, 3)
    resampled = dict(
        body_pose=np.stack([Slerp(source_times, Rotation.from_rotvec(pose[:, j]))(times).as_rotvec()
                            for j in range(21)], axis=1).reshape(len(times), 63),
        global_orient=Slerp(source_times, Rotation.from_rotvec(params["global_orient"]))(times).as_rotvec(),
        transl=np.column_stack([np.interp(times, source_times, params["transl"][:, d]) for d in range(3)]),
        betas=np.repeat(params["betas"].mean(0, keepdims=True), len(times), axis=0))
    model = smplx.create(str(ROOT.parents[1] / "repo/models/body_models"), model_type="smplx",
                         gender="neutral", flat_hand_mean=True, use_pca=False, batch_size=len(times)).eval()
    with torch.inference_mode():
        joints = model(**{k: torch.tensor(v, dtype=torch.float32) for k, v in resampled.items()}).joints[:, :22]
    os.chdir(REPO)
    features, canonical, _, _ = process_file(joints, .002)
    features = torch.tensor(features, dtype=torch.float32)
    _, root, angle = recover_root_rot_pos(features, return_rot_ang=True)
    features[:, 0], features[:, 1:4] = angle, root[:, [0, 2, 1]]
    recovered = recover_from_ric(features, 22, abs_3d=True).numpy()
    error = float(np.abs(recovered - canonical[:-1]).max())
    assert np.isfinite(features.numpy()).all() and error < 1e-4
    visible, _ = supported_points(state["observed_keypoints_2d"].numpy(),
                                  np.load(source / "input/target_masks.npz")["masks"],
                                  json.loads((ROOT / "protocol_v1.json").read_text())["visibility"])
    right = np.searchsorted(source_times, times).clip(0, n - 1)
    left = np.maximum(right - 1, 0)
    left[np.isclose(source_times[right], times, atol=1e-8)] = right[np.isclose(source_times[right], times, atol=1e-8)]
    observed = visible[left] & visible[right]
    joint_support = np.zeros((len(times), 22), dtype=bool)
    for j, name in enumerate(SMPL_NAMES):
        if name in COCO_NAMES:
            joint_support[:, j] = observed[:, COCO_NAMES.index(name)]
    # Full root frame is only treated as a supported estimate when both shoulders/hips are supported.
    root_support = observed[:, [5, 6, 11, 12]].all(1)
    joint_support[:, 0] = root_support
    mask = observation_mask(joint_support, root_support)
    mean = np.load(REPO / "dataset/HumanML3D_abs/Mean_abs_3d.npy")
    std = np.load(REPO / "dataset/HumanML3D_abs/Std_abs_3d.npy")
    condition = normalized_condition(features.numpy(), mask, mean, std)
    # Features at unknown coordinates are not persisted as model conditioning.
    np.savez_compressed(out / "condition.npz", condition=condition, observed_mask=mask, mean=mean, std=std,
                        time_20fps=times[:-1], source_times=source_times,
                        supported_joints=joint_support[:-1], estimated_source_joints=joints.numpy()[:-1],
                        estimated_canonical_joints=canonical[:-1],
                        original_body_pose=params["body_pose"], original_betas=params["betas"],
                        source_global_orient=resampled["global_orient"][:-1],
                        source_global_transl=resampled["transl"][:-1])
    report = dict(state="prepared_not_sampled", case=args.case, source=str(source),
                  native_motion_hash=hashlib.sha256((source / "motion/motion.safetensors").read_bytes()).hexdigest(),
                  estimated_pose_not_ground_truth=True, representation_roundtrip_max=error,
                  source_frames=n, representation_frames=len(condition),
                  unavailable_source_slots=np.flatnonzero(source_times > times[-2] + 1e-8).tolist(),
                  prompt=protocol["prompts"][sample], root_supported_frames=int(root_support[:-1].sum()),
                  observed_feature_elements=int(mask.sum()), hidden_contact_not_observed=True,
                  shape_retargeting="Official HumanML3D uniform_skeleton; generated joints require subject-specific SMPL-X fitting",
                  normalization="Official absolute-root; first-frame/floor normalization is model input convention, not Waymo canonical replacement",
                  limits=["3D conditions are estimates from occluded RGB, not true joints.",
                          "No hidden source reference image or reference detector output is read.",
                          "Official skeleton resizing is not identity-preserving; do not export this directly as the person.",
                          "Unrepresented terminal interval must remain explicitly unavailable."])
    (out / "preparation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
