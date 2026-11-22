"""Baseline YOLO/ByteTrack all-person candidates, one scene, no generation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "cache/ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(2)
    assert torch.cuda.device_count() == 1
    actual = str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
    assert actual == os.environ["FD_EXPECTED_GPU_UUID"].removeprefix("GPU-")
    cfg = json.loads((ROOT / "SCREEN_PROTOCOL.json").read_text())
    assert hashlib.sha256((ROOT / "SCREEN_PROTOCOL.json").read_bytes()).hexdigest() == (
        ROOT / "SCREEN_PROTOCOL.sha256").read_text().strip()
    row = next(r for r in json.loads((ROOT / "inventory.json").read_text())["scenes"]
               if r["split"] == args.split and r["scene"] == args.scene)
    out = ROOT / ("pilot" if args.pilot else "predicted_tracks") / args.split / args.scene
    out.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(Path(row["metadata"]).read_text())
    weights = Path(cfg["detector"]["weights"])
    assert weights.is_file(), "No implicit model download"
    model = YOLO(str(weights))
    started = time.time()
    cameras = []
    for cam in row["camera_indices"]:
        source = Path(row["processed_path"]) / "images"
        paths = [source / f"{i:03d}_{cam}.jpg" for i in range(row["frames"])]
        assert all(p.is_file() for p in paths), "Missing source frames"
        listing = out / f"camera{cam}.txt"
        listing.write_text("\n".join(map(str, paths)) + "\n")
        if model.predictor is not None and hasattr(model.predictor, "trackers"):
            for tracker in model.predictor.trackers:
                tracker.reset()
        histories = {}
        total_detections, untracked_detections, seen = 0, 0, 0
        results = model.track(source=str(listing), stream=True, persist=True,
            tracker=str(ROOT / "bytetrack.yaml"), device=0, conf=cfg["detector"]["conf"],
            imgsz=cfg["detector"]["imgsz"], classes=[0], batch=1, verbose=False,
            save=False, save_txt=False, save_crop=False, plots=False,
            project=str(ROOT / "cache/ultralytics_runs"), name="tracking", exist_ok=True)
        for index, result in enumerate(results):
            assert Path(result.path).resolve() == paths[index].resolve(), "Frame reorder"
            assert result.orig_shape == (1280, 1920), "Unsupported image geometry"
            seen += 1
            total_detections += len(result.boxes)
            if result.boxes.id is None:
                untracked_detections += len(result.boxes)
                continue
            ids = result.boxes.id.int().cpu().tolist()
            boxes = result.boxes.xyxy.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
            for track, box, score in zip(ids, boxes, scores):
                histories.setdefault(track, []).append({
                    "frame_index": index, "relative_time_s": metadata["normalized_time"][index],
                    "bbox_xyxy": box, "detector_confidence": score,
                    "timestamp_source": "processed_nominal_relative_time_not_camera_exposure"})
        assert seen == len(paths), (seen, len(paths))
        data = {"camera_index": int(cam), "frame_count": seen,
                "detected_boxes": total_detections, "untracked_boxes": untracked_detections,
                "tracks": [{"track_id": f"pred_cam{cam}_{track}", "observations": obs,
                            "associated_lidar_track_id": None,
                            "identity_scope": "predicted_camera_local_not_official"}
                           for track, obs in sorted(histories.items())]}
        (out / f"camera{cam}.json").write_text(json.dumps(data, allow_nan=False) + "\n")
        cameras.append({k: v for k, v in data.items() if k != "tracks"} |
                       {"track_count": len(histories)})
        listing.unlink()  # Frame list fully determined by metadata + source root.
    receipt = {"state": "tracked_not_visibility_or_identity_verified", "split": args.split,
        "scene": args.scene, "segment": row["segment"], "cameras": cameras,
        "seconds": time.time() - started, "pid": os.getpid(), "gpu_uuid": actual,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256((ROOT / "SCREEN_PROTOCOL.json").read_bytes()).hexdigest(),
        "tracker_sha256": hashlib.sha256((ROOT / "bytetrack.yaml").read_bytes()).hexdigest(),
        "source_metadata_sha256": hashlib.sha256(Path(row["metadata"]).read_bytes()).hexdigest(),
        "limitations": ["May miss people or split/mix tracks; no claim of official identities.",
            "Cross-camera duplicates have not yet been resolved.",
            "Source camera exposure timing is unavailable for this processed-data route."],
        "generation_started": False}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
