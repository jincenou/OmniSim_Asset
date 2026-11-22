"""Independent validation candidate; preserve reference geometry, save only useful G2 conditions."""
import argparse
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from safetensors.torch import load_file

G = Path(__file__).resolve().parent
W = G.parent
B = G / "validation12/generation"
REPO = Path(os.environ["FDANYONE_RUNTIME"])
sys.path[:0] = [str(REPO), str(W / "scene20/r20261019/scripts")]
from fdanyone.skeleton import pipeline as skeleton
from fdanyone.motion.result import MotionResult
from fdanyone.video import decode_canonical_clip
from fdanyone.assets import resolve_regressor, resolve_foreground_model
from fdanyone.views import resolve_view_plan
from geometry_precision import build
from root_local_geometry import separate_horizontal_root, source_camera_maps
from reliable_reference import reliable_reference


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, data):
    p.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample")
    sample = parser.parse_args().sample
    protocol = json.loads((B / "PROTOCOL.json").read_text())
    assert sha(B / "PROTOCOL.json") == (B / "PROTOCOL.sha256").read_text().strip()
    assert sample in protocol["subjects"]
    assert torch.cuda.device_count() == 1
    assert str(torch.cuda.get_device_properties(0).uuid) == os.environ["FD_EXPECTED_GPU_UUID"].removeprefix("GPU-")
    torch.set_num_threads(2)
    old = Path(protocol["sources"][sample])
    output = B / sample
    output.mkdir(exist_ok=False)
    source = old / "input/model_input.mp4"
    assert sha(source) == protocol["input_sha256"][sample]
    motion = MotionResult.load(old / "motion")
    n = motion.num_frames
    clip = decode_canonical_clip(source, num_frames=n, start_time=0, fps=Fraction(25))
    motion.validate_against_clip(clip)
    aligned_path = G / f"validation12/motion/cases/{sample}/text/aligned/motion_candidates.safetensors"
    receipt = json.loads((B / "receipts" / (sample + ".json")).read_text())
    assert sha(aligned_path) == receipt["aligned_sha256"]
    aligned = load_file(str(aligned_path))
    evidence = aligned["evidence_supported_visible_candidate"].numpy()
    native = {k: v.clone() for k, v in motion.smpl_params_global.items()}
    incam = {k: v.clone() for k, v in motion.smpl_params_incam.items()}
    native["betas"] = aligned["betas"].clone()
    incam["betas"] = aligned["betas"].clone()
    variants = {
        "G1": replace(motion, smpl_params_global=native, smpl_params_incam=incam),
        "G2": replace(
            motion,
            smpl_params_global=dict(native, body_pose=aligned["body_pose"]),
            smpl_params_incam=dict(incam, body_pose=aligned["body_pose"],
                                  global_orient=aligned["incam.global_orient"],
                                  transl=aligned["incam.transl"]),
        ),
    }
    geometry_fn = build(skeleton._body_geometry)
    regressor = resolve_regressor(None, model_dir=REPO / "models")
    geometries = {}
    maps_by_variant = {}
    reports = {}
    common_transform = None
    reference = None
    for label, variant in variants.items():
        geometry = geometry_fn(variant, regressor, REPO / "third_party/GVHMR", "cuda:0")
        if label == "G1":
            geometry, reference = reliable_reference(geometry, evidence)
            common_transform = geometry.motion_world_to_canonical_world.copy()
        else:
            transform = common_transform @ np.linalg.inv(geometry.motion_world_to_canonical_world)
            def transform_points(points):
                if points is None:
                    return None
                return (np.asarray(points, dtype=np.float64) @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)
            geometry = replace(
                geometry, vertices_world=transform_points(geometry.vertices_world),
                joints_world=transform_points(geometry.joints_world),
                keypoints_world=transform_points(geometry.keypoints_world),
                mesh_world=transform_points(geometry.mesh_world),
                motion_world_to_canonical_world=common_transform.copy())
        local, maps, checks = separate_horizontal_root(geometry)
        from validation_geometry import source_maps
        maps["local_to_smpl_incam"], rigid_error = source_maps(sys.modules[__name__], variant, geometry, local, maps["local_to_canonical"])
        geometries[label] = local
        maps_by_variant[label] = maps
        reports[label] = dict(root_checks=checks, source_geometry_check=rigid_error,
                              source_pose_scope="original" if label == "G1" else "CondMDI text plus O2 aligned pose and source root; native global root retained",
                              quality_approved=False)
    # Regenerate the same pinned BiRefNet masks; old scene20 cache was intentionally cleaned.
    masks_path = output / "source_masks.npz"
    masks = skeleton.predict_foreground_masks(clip.rgb_frames, resolve_foreground_model(REPO / "models"), "cuda:0")
    np.savez_compressed(masks_path, masks=masks)
    assert masks.shape == (n, clip.height, clip.width)
    union_points = np.concatenate([g.keypoints_world for g in geometries.values()], axis=0)
    common_center = geometries["G1"].vertices_world.mean(axis=(0, 1))
    common_center[1] = 0
    original_ring, original_grid = skeleton.camera_ring, skeleton.camera_grid
    original_framing = skeleton.solve_clip_framing
    framing_cache = {}

    def common_camera(factory, **kwargs):
        center = common_center.copy()
        center[1] = kwargs["target_height"]
        kwargs.update(center=center, front_direction=np.array([0., 0., -1.]))
        return factory(**kwargs)

    skeleton.camera_ring = lambda **kw: common_camera(original_ring, **kw)
    skeleton.camera_grid = lambda **kw: common_camera(original_grid, **kw)
    def framing(points, names, profile, camera_factory, aspect_ratio):
        if framing_cache:
            return framing_cache["value"]
        base = original_framing(union_points, names, profile, camera_factory, aspect_ratio)
        def inside(radius):
            for camera in camera_factory(radius, base.target_height):
                crop = skeleton.center_crop(camera.image_height, camera.image_width, 1280, 704)
                K = np.asarray(camera.K).copy()
                K[0, 0] = K[1, 1] = base.focal_normalized * camera.image_height
                camera = skeleton._cropped_camera(replace(camera, K=tuple(map(tuple, K))), crop)
                world = np.asarray(camera.world_to_camera)
                p = union_points @ world[:3, :3].T + world[:3, 3]
                q = p @ np.asarray(camera.K).T
                uv = q[..., :2] / q[..., 2:]
                if not ((p[..., 2] > 0) & (uv[..., 0] >= .06001 * camera.image_width) &
                        (uv[..., 0] <= .93999 * camera.image_width) &
                        (uv[..., 1] >= .06001 * camera.image_height) &
                        (uv[..., 1] <= .93999 * camera.image_height)).all():
                    return False
            return True
        low, high = .1 * base.radius, base.radius
        while not inside(high):
            high *= 1.15
            if high > 2 * base.radius:
                raise RuntimeError("Common all-time framing exceeds fixed radius limit")
        for _ in range(24):
            mid = (low + high) / 2
            if inside(mid):
                high = mid
            else:
                low = mid
        framing_cache["value"] = replace(base, radius=high, radius_solve=replace(
            base.radius_solve, radius=high, bound=None, limiting_constraint="paired_all_time_joint_bounds"))
        return framing_cache["value"]
    skeleton.solve_clip_framing = framing
    skeleton.predict_foreground_masks = lambda *args, **kw: masks.copy()
    camera_payloads, metadata = {}, {}
    for label in ("G2",):
        variant = variants[label]
        root = output / label
        (root / "inputs").mkdir(parents=True)
        (root / "preflight").mkdir()
        (root / "inputs/main.mp4").symlink_to(source)
        # MotionResult.save preserves CPU storage; give each serialized key independent storage.
        serialized = replace(variant,
                             smpl_params_global={k: v.clone() for k, v in variant.smpl_params_global.items()},
                             smpl_params_incam={k: v.clone() for k, v in variant.smpl_params_incam.items()})
        serialized.save(root / "preflight/motion")
        np.savez_compressed(root / "preflight/root_motion.npz", **maps_by_variant[label],
                            frame_times=np.asarray(motion.frame_timestamps_sec))
        skeleton._body_geometry = lambda *args, **kw: geometries[label]
        target = root / "preflight/conditioning"
        skeleton.build_skeleton_conditioning(
            motion=variant, clip=clip, regressor_path=regressor,
            foreground_model_path=resolve_foreground_model(REPO / "models"),
            gvhmr_root=REPO / "third_party/GVHMR", output_dir=target, device="cuda:0",
            view_plan=resolve_view_plan(views_per_layer=24, layer_pitches=[15],
                                       start_yaw=0, yaw_span=360, enable_rcp=True, enable_tcr=True))
        payload = json.loads((target / "cameras.json").read_text())
        payload["original_motion_world_to_canonical_world"] = payload.pop("motion_world_to_canonical_world")
        payload["world_frame"] = dict(payload["world_frame"], name="root_local_human_world")
        payload["motion_world_to_local_world_file"] = "../root_motion.npz"
        save(target / "cameras.json", payload)
        camera_payloads[label] = payload
        metadata[label] = json.loads((target / "metadata.json").read_text())
        crop = metadata[label]["source_crop"]
        binary = masks > 12.75
        retained = binary[:, crop["top"]:crop["top"] + crop["height"],
                          crop["left"]:crop["left"] + crop["width"]].sum((1, 2)) / np.maximum(binary.sum((1, 2)), 1)
        assert retained.min() >= .95
        reports[label]["source_mask_retained_min"] = float(retained.min())
    # Both geometries define the same Dev6 framing rule; only G2 is serialized.
    # Legacy G0 was rendered with its own cameras: do not claim identical old/new cameras.
    from fdanyone.skeleton.pipeline import Conditioning
    for label in ("G2",):
        Conditioning.load(output / label / "preflight/conditioning")
    paired = {
        "sample": sample, "frames": n, "views": 24, "reference": reference,
        "camera_arrays_identical_to_old_G0": False, "source_crop_identical_to_old_G0": None,
        "source_sha256": sha(source), "aligned_sha256": sha(aligned_path),
        "cached_mask_sha256": sha(masks_path), "protocol_sha256": sha(B / "PROTOCOL.json"),
        "reports": reports, "quality_approved": False,
        "limitations": protocol["limitations"],
    }
    save(output / "pairing.json", paired)
    for label in ("G2",):
        save(output / label / "preflight/review.json", {
            "allow_generation": True, "quality_approved": False,
            "scope": "Fixed independent validation candidate after geometry checks; known Dev failures remain, not adoption",
            "pairing": str(output / "pairing.json")})
    print(json.dumps({"sample": sample, "state": "candidate_conditioning_verified_not_quality_approved"}), flush=True)


if __name__ == "__main__":
    main()
