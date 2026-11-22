"""Build a time-indexed context window without inventing boundary observations."""
import json
from pathlib import Path
import sys
import hashlib
import av
import numpy as np
from PIL import Image
from safetensors.torch import load_file, save_file
import torch

G = Path(__file__).resolve().parent
B = G / "text_motion/bilateral_context"
E = B / "repair_v1"
sys.path[:0] = [str(G.parent / "frontend_quality_20261020"), str(G.parent / "scripts")]
from motion_evidence import resample_measured_camera
from prepare import encode


def main():
    full = B / "P18/calibrated/input"
    original = G.parent / "scene20/r20261019/scenes/005/P18"
    support = np.load(B / "P18/support.npz")["supported"]
    rows = json.loads((full / "inputs/frame_mapping.json").read_text())["frames"]
    native = np.load(full / "native_cameras.npz")
    nt = native["image_center_timestamps_s"]
    times = np.array([r["source_relative_exposure_seconds"] for r in rows])
    target_cam = np.load(original / "input/native_cameras.npz")
    target_rows = json.loads((original / "input/frame_mapping.json").read_text())["frames"]
    target_times = (target_cam["image_center_timestamps_s"][0] - nt[0]
                    + np.array([r["source_relative_exposure_seconds"] for r in target_rows]))
    root = support[:, [5, 6, 11, 12]].all(1)
    # An engineering admission rule, not a claim that supported 3D is correct.
    candidates = [i for i in range(len(root)-5)
                  if root[i:i+6].all() and times[i] < target_times[0]]
    if not candidates:
        raise ValueError("No supported pre-gap root span; cannot fabricate one")
    start = candidates[0]
    assert start == 25  # This run's frozen source selection, checked before sampling.
    E.mkdir(exist_ok=False)
    source = E / "source"
    (source / "motion").mkdir(parents=True)
    (source / "input").mkdir()
    state = load_file(str(full / "motion_pair/calibrated_moving/motion.safetensors"))
    trimmed = {k: v[start:].clone() for k, v in state.items()}
    identity = load_file(str(original / "motion/motion.safetensors"))
    beta = identity["smpl_params_incam.betas"].mean(0, keepdim=True)
    for group in ("smpl_params_incam", "smpl_params_global"):
        trimmed[group + ".betas"] = beta.expand(len(rows)-start, -1).clone()
    save_file(trimmed, str(source / "motion/motion.safetensors"))
    masks = np.load(full / "context_segmentation_points/target_masks.npz")["masks"][start:]
    np.savez_compressed(source / "input/target_masks.npz", masks=masks)
    selected_rows = []
    for index, row in enumerate(rows[start:]):
        selected_rows.append(dict(row, slot=index, video_seconds=index/25.,
                                  full_context_slot=index+start,
                                  source_relative_exposure_seconds=row["source_relative_exposure_seconds"]-times[start]))
    (source / "input/frame_mapping.json").write_text(json.dumps({
        "frames": selected_rows, "full_context": str(full),
        "relative_origin_exposure_s": float(nt[0]+times[start]),
        "source_span_seconds": float(times[-1]-times[start]),
    }, indent=2) + "\n")
    camera = resample_measured_camera(nt-nt[0], native["image_center_cv_camera_to_world"], times[start:])
    np.savez_compressed(source / "input/measured_cameras.npz",
                        camera_to_waymo_world=camera, relative_exposure_seconds=times[start:]-times[start])
    with av.open(str(full / "inputs/main.mp4")) as con:
        con.streams.video[0].thread_count = 1
        rgb = [f.to_ndarray(format="rgb24") for f in con.decode(video=0)][start:]
    assert len(rgb) == len(masks) == 76
    encode(source / "input/model_input.mp4", [Image.fromarray(im) for im in rgb], 25)
    with av.open(str(source / "input/model_input.mp4")) as con:
        con.streams.video[0].thread_count = 1
        decoded = [f.to_ndarray(format="rgb24") for f in con.decode(video=0)]
    assert len(decoded) == len(rgb) and all(np.array_equal(a, b) for a, b in zip(rgb, decoded))
    evidence = {
        "state": "context_window_prepared_not_completed_motion",
        "source": str(source), "full_context": str(full), "original_target": str(original),
        "selection": "Earliest six consecutive shoulder/hip-supported25fps slots before original target start",
        "support_is_estimate_not_truth": True,
        "full_context_slot_range": [start, len(rows)-1],
        "frames": len(rgb), "cfr_seconds": (len(rgb)-1)/25.,
        "relative_origin_exposure_s": float(nt[0]+times[start]),
        "target_times_in_context_seconds": (target_times-times[start]).tolist(),
        "original_target_frames": len(target_rows),
        "target_window_covered": bool((target_times >= times[start]).all() and (target_times <= times[-1]).all()),
        "betas": "Fixed mean of original P18 identity, not re-estimated identity shape",
        "beta_sha256": hashlib.sha256(beta.numpy().tobytes()).hexdigest(),
        "pixels_verified_after_encoding": True,
        "input_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            full / "motion_pair/calibrated_moving/motion.safetensors",
            B / "P18/support.npz", full / "context_segmentation_points/target_masks.npz",
            original / "motion/motion.safetensors")},
    }
    (E / "source.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({k: evidence[k] for k in ("frames", "full_context_slot_range", "target_window_covered")}))


if __name__ == "__main__":
    main()
