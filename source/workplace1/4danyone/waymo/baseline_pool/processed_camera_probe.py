"""Validate processed camera geometry without inventing exposure timestamps."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from scan_native import ROOT, write, sha


def camera_sequence(row, camera, observations):
    scene = Path(row["processed_path"])
    meta = json.loads(Path(row["metadata"]).read_text())
    intrinsic = np.loadtxt(scene / "intrinsics" / f"{camera}.txt").reshape(-1)
    assert intrinsic.shape == (9,) and np.isfinite(intrinsic).all()
    assert min(intrinsic[:2]) > 0
    ext = np.loadtxt(scene / "cam_to_ego" / f"{camera}.txt")
    axes = np.eye(4)
    axes[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    K = np.array([[intrinsic[0], 0, intrinsic[2]], [0, intrinsic[1], intrinsic[3]], [0, 0, 1]])
    matrices, times, errors, records = [], [], [], []
    for observation in observations:
        index = observation["frame_index"]
        ego_path = scene / "ego_to_world" / f"{index:03d}.txt"
        camera_path = scene / "cam_to_world" / f"{index:03d}_{camera}.txt"
        ego, stored = np.loadtxt(ego_path), np.loadtxt(camera_path)
        error = float(np.abs(ego @ ext - stored).max())
        assert error < 1e-5, "Processed camera composition mismatch"
        transform = stored @ axes
        rotation = transform[:3, :3]
        assert np.max(np.abs(rotation @ rotation.T - np.eye(3))) < 1e-5
        assert abs(np.linalg.det(rotation) - 1) < 1e-5
        pixels = scene / "images" / f"{index:03d}_{camera}.jpg"
        with Image.open(pixels) as im:
            assert im.size == (1920, 1280)
        matrices.append(transform)
        times.append(meta["normalized_time"][index])
        errors.append(error)
        records.append({"frame_index": index, "source_path": str(pixels),
                        "source_sha256": sha(pixels), "camera_matrix_sha256": sha(camera_path),
                        "ego_matrix_sha256": sha(ego_path), "bbox_xyxy": observation["bbox_xyxy"]})
    times = np.asarray(times, dtype=float)
    assert np.all((np.diff(times) >= .08) & (np.diff(times) <= .12))
    return {"K": K, "distortion": intrinsic[4:], "cv_camera_to_world": np.array(matrices),
            "R_world_to_cv_camera": np.array(matrices)[:, :3, :3].transpose(0, 2, 1),
            "nominal_relative_time_s": times}, {
                "camera_composition_max_abs": max(errors), "records": records,
                "source_metadata_sha256": sha(Path(row["metadata"])),
                "timing": "Processed nominal frame time; original image exposure timing unavailable.",
                "pose": "Frame ego pose times camera extrinsic; not exposure-center camera pose.",
                "raw_pixel_equality": "unverified_without_raw",
                "generation_allowed": False}


def main():
    rows = {(r["split"], r["scene"]): r
            for r in json.loads((ROOT / "inventory.json").read_text())["scenes"]}
    pilots = json.loads((ROOT / "pilot_manifest.json").read_text())["candidates"]
    comparison = []
    for candidate in pilots:
        arrays, report = camera_sequence(rows[candidate["split"], candidate["scene"]],
                                         candidate["camera_index"], candidate["observations"])
        native = np.load(ROOT / "camera_audit" / candidate["review_id"] / "native_cameras.npz")
        np.testing.assert_allclose(arrays["K"], native["K"], atol=1e-5, rtol=0)
        np.testing.assert_allclose(arrays["distortion"], native["distortion"], atol=1e-5, rtol=0)
        a, b = arrays["cv_camera_to_world"], native["image_center_cv_camera_to_world"]
        angles = Rotation.from_matrix(a[:, :3, :3] @ b[:, :3, :3].transpose(0, 2, 1)).magnitude()
        native_times = native["image_center_timestamps_s"]
        comparison.append({"sample": candidate["review_id"],
                           "camera_rotation_difference_degrees_max": float(np.rad2deg(angles).max()),
                           "camera_translation_difference_m_max": float(np.linalg.norm(a[:, :3, 3] - b[:, :3, 3], axis=1).max()),
                           "relative_time_difference_s_max": float(np.abs(
                               arrays["nominal_relative_time_s"] - arrays["nominal_relative_time_s"][0] -
                               (native_times-native_times[0])).max()),
                           "processed_composition_max_abs": report["camera_composition_max_abs"]})
    write(ROOT / "research/processed_camera_probe.json", {
        "comparisons": comparison, "scope": "Four fixed native pilot windows only, not full dataset.",
        "interpretation": "Measures approximation difference, not motion/pose accuracy.",
        "processed_only_pipeline_admitted": False})
    print(json.dumps(comparison), flush=True)


if __name__ == "__main__":
    main()
