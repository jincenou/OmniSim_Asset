"""Restore the old retained conditioning and reuse the new fixed conditioning."""
import argparse
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

G = Path(__file__).resolve().parent
V = G / "validation12"
S = V / "second_seed"
sys.path.insert(0, os.environ["FDANYONE_RUNTIME"])


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample")
    args = parser.parse_args()
    cfg = read(S / "PROTOCOL.json")
    assert args.sample in cfg["subjects"]
    source = Path(cfg["sources"][args.sample])
    parent = V / "generation" / args.sample / "G2"
    from fdanyone.motion.result import MotionResult
    from fdanyone.skeleton.pipeline import Conditioning
    from fdanyone.geometry.crop import center_crop, crop_from_bounds, mask_bounds
    from fdanyone.video import decode_canonical_clip, verify_lossless_video
    from fdanyone.views import ViewPlan

    old = S / args.sample / "G0"
    old.mkdir(parents=True, exist_ok=False)
    (old / "inputs").mkdir()
    (old / "inputs/main.mp4").symlink_to(source / "input/model_input.mp4")
    (old / "preflight").mkdir()
    (old / "preflight/motion").symlink_to(source / "motion", target_is_directory=True)
    (old / "preflight/root_motion.npz").symlink_to(source / "root_motion.npz")
    condition = old / "preflight/conditioning"
    condition.mkdir()
    (condition / "goliath40").symlink_to(source / "skeletons", target_is_directory=True)
    (condition / "source.mkv").symlink_to(parent / "preflight/conditioning/source.mkv")
    metadata = read(parent / "preflight/conditioning/metadata.json")
    generation = read(source / "generation.json")
    assert generation["preprocessing"]["source_crop_policy"] == metadata["source_crop_policy"]
    assert generation["preprocessing"]["foreground_model"] == metadata["foreground_model"]
    frames = cfg["frames"][args.sample]
    clip = decode_canonical_clip(old / "inputs/main.mp4", num_frames=frames, start_time=0, fps=Fraction(25))
    MotionResult.load(old / "preflight/motion").validate_against_clip(clip)
    verify_lossless_video(clip, condition / "source.mkv")
    policy = generation["preprocessing"]["source_crop_policy"]
    masks = np.load(parent.parent / "source_masks.npz")["masks"]
    rebuilt_crop = crop_from_bounds(
        bounds=mask_bounds(masks, policy["threshold"]), image_height=clip.height,
        image_width=clip.width, output_height=1280, output_width=704,
        margins=tuple(policy["margins"]), allow_upscale=policy["allow_upscale"])
    assert asdict(rebuilt_crop) == metadata["source_crop"]
    target_crop = asdict(center_crop(clip.height, clip.width, 1280, 704))
    cameras = read(source / "cameras.json")
    for camera in cameras["cameras"]:
        camera["pitch_degrees"] = camera.pop("pitch")
        camera["yaw_degrees"] = camera.pop("yaw")
        camera["crop"] = target_crop
        camera["world_to_camera"] = np.linalg.inv(np.asarray(camera["camera_to_world"])).tolist()
        camera["skeleton_video"] = f'goliath40/{camera["camera_id"]:02d}.mp4'
    plan = ViewPlan.from_dict(metadata["view_plan"])
    assert plan.is_canonical_target_ring
    cameras["rcp_cameras"] = [cameras["cameras"][v].copy() for v in plan.rcp_camera_ids]
    for name in ("framing", "skeleton_draw_scale"):
        metadata[name] = generation["preprocessing"][name]
    save(condition / "metadata.json", metadata)
    save(condition / "cameras.json", cameras)
    restored = Conditioning.load(condition)
    assert restored.num_frames == frames and len(restored.target_skeletons) == 24
    # These are the original pre-network skeleton bitstreams, copied by output_writer.
    manifest = read(source / "manifest.json")
    retained = manifest["retained"]
    for path in [source / "skeletons" / f"{v:02d}.mp4" for v in range(24)]:
        assert sha(path) == retained[str(path.relative_to(source))]
    save(old / "preflight/review.json", {
        "allow_generation": True, "quality_approved": False,
        "scope": "Fixed second seed; original retained skeleton/camera/motion, source policy reconstructed.",
    })
    new = old.parent / "G2"
    new.mkdir()
    (new / "inputs").symlink_to(parent / "inputs", target_is_directory=True)
    (new / "preflight").symlink_to(parent / "preflight", target_is_directory=True)
    paths = [source / "generation.json", source / "cameras.json", source / "motion/motion.safetensors",
             source / "motion/motion.json", source / "input/model_input.mp4",
             parent.parent / "source_masks.npz"]
    paths += sorted((condition).rglob("*.json"))
    paths += sorted((source / "skeletons").glob("*.mp4"))
    paths += [parent / "preflight/conditioning/metadata.json",
              parent / "preflight/conditioning/cameras.json",
              parent / "preflight/conditioning/source.mkv",
              parent / "preflight/motion/motion.json",
              parent / "preflight/motion/motion.safetensors"]
    paths += sorted((parent / "preflight/conditioning/goliath40").glob("*.mp4"))
    save(old.parent / "conditioning_audit.json", {
        "state": "both_conditioning_sets_prepared",
        "sample": args.sample, "seed": cfg["seed"], "frames": frames,
        "old_skeleton_bitstreams_match_original_manifest": True,
        "old_camera_matrices_reused": True, "old_motion_reused": True,
        "source_pixels_losslessly_equal_model_input": True,
        "source_crop_recomputed_same_pinned_model_and_policy": True,
        "old_original_source_crop_bitwise_proof": None,
        "new_G2_conditions_exactly_reused": True,
        "quality_approved": False,
        "limitation": "Legacy source crop/mask cache was removed before this study. Crop is reconstructed from the pinned original foreground model/policy, not a claim of bitwise verification against that missing cache.",
        "input_sha256": {str(p): sha(p) for p in paths},
    })
    print(json.dumps({"sample": args.sample, "state": "prepared_not_generated"}))


if __name__ == "__main__":
    main()
