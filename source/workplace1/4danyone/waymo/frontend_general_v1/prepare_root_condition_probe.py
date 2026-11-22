"""Freeze a paired diagnostic before sampling; no changes to prior experiments."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from safetensors.torch import load_file, save_file
import torch
from optimize_motion import supported_points

G = Path(__file__).resolve().parent
OLD = G / "text_motion/bilateral_context/repair_v1"
E = OLD.parent / "root_condition_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(name, obj):
    path = E / (name + ".json")
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    path.with_suffix(".sha256").write_text(sha(path) + "\n")


def main():
    E.mkdir(exist_ok=False)
    (E / "source/motion").mkdir(parents=True)
    (E / "source/input").symlink_to(OLD / "source/input", target_is_directory=True)
    (E / "box_prior.npz").symlink_to(OLD / "box_prior.npz")
    state = load_file(str(OLD / "source/motion/motion.safetensors"))
    original = {k: v.clone() for k, v in state.items()}
    cfg = json.loads((OLD / "BOX_PROTOCOL.json").read_text())
    visible, _ = supported_points(
        state["observed_keypoints_2d"].numpy(),
        np.load(OLD / "source/input/target_masks.npz")["masks"], cfg["visibility"])
    box = np.load(OLD / "box_prior.npz")
    cameras = np.load(OLD / "source/input/measured_cameras.npz")
    np.testing.assert_allclose(box["times_seconds"], cameras["relative_exposure_seconds"], rtol=0, atol=1e-8)
    trusted = visible[:, [5, 6, 11, 12]].all(1) & box["supported"]
    assert trusted.sum() >= 6
    global_r = Rotation.from_rotvec(state["smpl_params_global.global_orient"].numpy()).as_matrix()
    incam_r = Rotation.from_rotvec(state["smpl_params_incam.global_orient"].numpy()).as_matrix()
    maps = cameras["camera_to_waymo_world"][:, :3, :3] @ incam_r @ global_r.transpose(0, 2, 1)
    # Rotation estimates only establish one fixed gravity-frame alignment.
    # Annotation heading and per-frame velocity never set body orientation.
    fixed = Rotation.from_matrix(maps[trusted]).mean()
    residual = (fixed.inv() * Rotation.from_matrix(maps)).magnitude()
    assert np.isfinite(residual).all() and np.max(residual) < np.deg2rad(10)
    assert (fixed.as_matrix() @ np.array([0., 1., 0.]))[2] > .98
    displacement = box["box_center_world"] - box["box_center_world"][:1]
    gravity_displacement = displacement @ fixed.as_matrix()
    old_trans = original["smpl_params_global.transl"].numpy()
    new_trans = old_trans.copy()
    new_trans[:, [0, 2]] = old_trans[:1, [0, 2]] + gravity_displacement[:, [0, 2]]
    state["smpl_params_global.transl"] = torch.from_numpy(new_trans)
    # No pose, heading, identity, camera-space state or image evidence changes.
    for key in original:
        if key != "smpl_params_global.transl":
            assert torch.equal(original[key], state[key]), key
    np.testing.assert_array_equal(new_trans[:, 1], old_trans[:, 1])
    np.testing.assert_array_equal(new_trans[0], old_trans[0])
    assert np.isfinite(new_trans).all()
    save_file({k: v.clone().contiguous() for k, v in state.items()},
              str(E / "source/motion/motion.safetensors"))
    report = dict(
        state="source_frozen_before_sampling", quality_approved=False,
        variant="Waymo 3D-box-assisted horizontal root conditioning",
        annotation_heading_used=False, local_pose_and_orientation_bitwise_preserved=True,
        camera_space_state_bitwise_preserved=True, original_root_height_preserved=True,
        trusted_rotation_observations=int(trusted.sum()),
        fixed_gravity_to_world_rotation=fixed.as_matrix().tolist(),
        alignment_residual_degrees_median_max=np.degrees([np.median(residual), residual.max()]).tolist(),
        original_horizontal_path_m=float(np.linalg.norm(np.diff(old_trans[:, [0, 2]], axis=0), axis=1).sum()),
        new_horizontal_path_m=float(np.linalg.norm(np.diff(new_trans[:, [0, 2]], axis=0), axis=1).sum()),
        time_slots=len(new_trans),
        limits="Box center is not pelvis or hidden joints. Translation-only condition change; old root-support masks unchanged. The fixed global/world frame alignment remains estimated.",
    )
    (E / "source_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    protocol = json.loads((OLD / "PROTOCOL.json").read_text())
    protocol.update(
        frozen_utc_seconds=time.time(),
        stage="Paired root-conditioning diagnostic; two samples only, no 24-view generation",
        source_branch="Same expanded real context; 3D-box-assisted horizontal global translation",
        sources={"P18": str(E / "source")},
        comparators=["repair_v1/no_text/box_assisted", "repair_v1/text/box_assisted"],
        selection="Exactly no_text/text, seed42; retain both, no best selection",
        comparison="Only pre-prior horizontal global root translation changes; same images, local rotations, masks, sampler, retargeting and box-assisted alignment",
        counterfactual_results_seen=True,
        recipe_changes_after_counterfactual="Explicit new 3D-supervised root-conditioning diagnostic, not a claim of unchanged monocular protocol",
    )
    files = [Path(p) for p in protocol["code_and_source_sha256"]]
    files += [G / name for name in (
        "root_condition_probe.py", "prepare_root_condition_probe.py",
        "run_root_condition_probe.py", "align_bilateral_boxes.py")]
    files += [E / "source/motion/motion.safetensors", E / "box_prior.npz", E / "source_audit.json"]
    protocol["code_and_source_sha256"] = {str(p): sha(p) for p in files}
    freeze("PROTOCOL", protocol)
    retarget = json.loads((OLD / "RETARGET_PROTOCOL.json").read_text())
    retarget["source_protocol"] = str(E / "PROTOCOL.json")
    freeze("RETARGET_PROTOCOL", retarget)
    cfg["code_sha256"] = {str(G / "align_bilateral_boxes.py"): sha(G / "align_bilateral_boxes.py"),
                          str(E / "box_prior.npz"): sha(E / "box_prior.npz")}
    cfg["comparison_scope"] = "Same box-assisted alignment; only upstream root conditioning differs"
    freeze("BOX_PROTOCOL", cfg)
    (E / "PLAN.md").write_text(
        "# 根部运动条件配对诊断\n\n"
        "2026-10-20，正式采样前冻结。沿用repair_v1的76时刻、真实前后文、可见性、"
        "原始局部姿态与朝向，只把送入CondMDI的global水平位移替换为关联Waymo框相对位移。"
        "以可靠肩髋时刻的相机/根旋转确定一个固定坐标变换，保留原根高度，不使用框朝向。\n\n"
        "两分支仍为无文字/有文字，同seed42、1000采样步、200步重定向和100+200步框辅助对齐。"
        "与repair_v1同名box_assisted比较；不与纯单目方案混称单变量。"
        "原始图像、姿态、体型、掩码均相同，检查实际条件mask和初始噪声一致。\n\n"
        "先比较完整时间骨架、两端可见动作及缺口衔接；检测NME、加速度和框轨迹误差均不能证明隐藏动作真实。"
        "无可信脚接触或隐藏姿态真值，不设膝角上限，不锁脚，不选seed，不启动新的24路视频。\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
