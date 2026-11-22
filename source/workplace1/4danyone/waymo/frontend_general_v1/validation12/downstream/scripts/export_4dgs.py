"""Freeze existing FTGS cameras and RGBA inputs for an independent backend recipe."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

R = Path(__file__).resolve().parents[1]
A = R / "four_dg"
sys.path[:0] = [str(R / "backend")]
from datasets.FreeTime_dataset import FreeTimeParser
from datasets.normalize import transform_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    out = A / "data" / args.label
    out.mkdir(parents=True, exist_ok=False)
    with np.load(args.data / "init.npz") as values:
        denominator = int(values["time_denominator"])
        frames = denominator + 1
        assert frames >= 13
        assert int(values["frame_start"]) == 0 and int(values["frame_end"]) == frames
        source = FreeTimeParser(str(args.data), normalize=True, start_frame=0, end_frame=frames)
        assert len(source.camera_names) == 24 and source.total_frames == frames
        all_points = transform_points(source.transform, values["positions"])
        at_zero = np.isclose(values["times"], 0, atol=1e-7)
        assert at_zero.sum() > 100
        canonical = all_points[at_zero]
        colors = np.clip(values["colors"][at_zero] / 255.0, 0, 1)
        np.savez(out / "canonical.npz", points=canonical, colors=colors)
    cameras = []
    holdout = [0, 8, 16]
    hashes = []
    for index, cid in enumerate(source.camera_ids):
        view = int(cid) - 1
        assert source.camera_names[index] == f"cam{view:02d}"
        width, height = source.imsize_dict[cid]
        assert (width, height) == (352, 640)
        K = source.Ks_dict[cid]
        assert np.allclose(K[0, 2], width / 2) and np.allclose(K[1, 2], height / 2)
        cameras.append({"view": view, "K": K.tolist(), "camtoworld": source.camtoworlds[index].tolist()})
        assert sorted(int(path.stem) for path in (args.data / "images" / f"cam{view:02d}").glob("*.png")) == list(range(frames))
        for frame in range(frames):
            path = args.data / "images" / f"cam{view:02d}" / f"{frame:06d}.png"
            with Image.open(path) as image:
                assert image.size == (width, height) and image.mode == "RGBA"
            hashes.append({"view": view, "frame": frame, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    assert sorted(c["view"] for c in cameras) == list(range(24))
    low, high = all_points.min(0), all_points.max(0)
    margin = np.maximum((high - low) * .1, .02)
    centers = source.camtoworlds[:, :3, 3]
    radius = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)
    manifest = {"version": 1, "label": args.label, "source_data": str(args.data.resolve()),
        "frame_count": frames, "time_denominator": denominator, "fps": 25, "width": 352, "height": 640,
        "sample_span_seconds": denominator / 25,
        "train_views": [view for view in range(24) if view not in holdout], "heldout_views": holdout,
        "cameras": cameras, "canonical_npz": "canonical.npz", "initial_points": int(len(canonical)),
        "scene_radius": radius, "source_to_normalized": source.transform.tolist(),
        "motion_aabb_min": (low - margin).tolist(), "motion_aabb_max": (high + margin).tolist(),
        "initialization": "Only time0 surface points; all-time training-init extent supplies deformation bounds. No heldout RGBA used to initialize geometry.",
        "source_init_sha256": hashlib.sha256((args.data / "init.npz").read_bytes()).hexdigest(),
        "input_hashes": hashes, "comparison_scope": "Complete backend recipe; initialization/loss differ from FTGS."}
    (out / "fdanyone.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"label": args.label, "points": len(canonical), "images": len(hashes), "out": str(out)}))


if __name__ == "__main__":
    main()
