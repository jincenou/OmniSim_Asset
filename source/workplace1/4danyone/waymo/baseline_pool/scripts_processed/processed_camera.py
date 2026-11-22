"""Processed-only calibrated camera adapter; no fabricated raw observations."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from processed_camera_probe import camera_sequence
from scan_native import write, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--review-id", required=True)
    args = parser.parse_args()
    candidate = next(c for c in json.loads(args.manifest.read_text())["candidates"]
                     if c["review_id"] == args.review_id)
    row = next(r for r in json.loads((ROOT / "inventory.json").read_text())["scenes"]
               if r["split"] == candidate["split"] and r["scene"] == candidate["scene"])
    assert sha(Path(candidate["track_cache"])) == candidate["track_cache_sha256"]
    arrays, report = camera_sequence(row, candidate["camera_index"], candidate["observations"])
    output = ROOT / "camera_audit" / candidate["review_id"]
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for rec, t in zip(report["records"], arrays["nominal_relative_time_s"], strict=True):
        records.append({"frame_index": rec["frame_index"], "frame_timestamp_micros": None,
                        "sample_timestamp_s": float(t), "timestamp_basis": "processed_nominal_relative",
                        "source_path": rec["source_path"], "processed_jpeg_sha256": rec["source_sha256"],
                        "bbox_xyxy_distorted": rec["bbox_xyxy"],
                        "raw_pixel_equality": "unverified_without_raw",
                        "distortion_provenance": "stored nine-parameter intrinsics; same processed export convention checked on four native pilot windows"})
    np.savez_compressed(output / "native_cameras.npz",
                        K=arrays["K"], distortion=arrays["distortion"],
                        frame_cv_camera_to_world=arrays["cv_camera_to_world"],
                        R_world_to_cv_camera=arrays["R_world_to_cv_camera"],
                        sample_timestamps_s=arrays["nominal_relative_time_s"])
    write(output / "audit.json", {
        "state": "processed_camera_composition_checked_exposure_unknown",
        "scene": candidate["scene"], "track_id": candidate["track_id"],
        "camera_index": candidate["camera_index"], "records": records,
        "camera_composition_max_abs": report["camera_composition_max_abs"],
        "source_metadata_sha256": report["source_metadata_sha256"],
        "limitations": [report["timing"], report["pose"],
                       "No original JPEG/exposure comparison for this scene.",
                       "Does not establish motion accuracy or cross-camera identity."],
        "generation_allowed": False})
    print(json.dumps({"state": "processed_camera_ready_for_pilot", "frames": len(records)}))


if __name__ == "__main__":
    main()
