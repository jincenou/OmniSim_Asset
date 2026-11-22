"""Incremental candidate screen. Camera tracks are NOT deduplicated people."""
import argparse
import asyncio
import collections
import json
import os
from pathlib import Path
import time

import numpy as np

from scan_native import ROOT, sha, write


def select_windows(observations, cfg):
    obs = sorted(observations, key=lambda x: x["frame_index"])
    frames = np.array([o["frame_index"] for o in obs])
    if len(set(frames)) != len(frames):
        raise ValueError("Duplicate frame in track")
    groups = np.split(np.arange(len(obs)), np.flatnonzero(np.diff(frames) != 1) + 1)
    rejected, ranked = collections.Counter(), []
    for length in cfg["native_lengths"]:
        for group in groups:
            for offset in range(max(0, len(group) - length + 1)):
                part = [obs[int(i)] for i in group[offset:offset + length]]
                boxes = np.asarray([o["bbox_xyxy"] for o in part])
                heights, widths = boxes[:, 3] - boxes[:, 1], boxes[:, 2] - boxes[:, 0]
                if not np.isfinite(boxes).all() or np.any(heights <= 0) or np.any(widths <= 0):
                    rejected["invalid_box"] += 1
                    continue
                if heights.min() < cfg["minimum_height_px"] or np.median(heights) < cfg["median_height_px"]:
                    rejected["resolution"] += 1
                    continue
                if np.any(boxes[:, :2] < 0) or np.any(boxes[:, 2:] > [1920, 1280]):
                    rejected["outside_image"] += 1
                    continue
                dt = np.diff([o["relative_time_s"] for o in part])
                if np.any((dt < .08) | (dt > .12)):
                    rejected["timestamp_gap"] += 1
                    continue
                # Confidence is a detector score, never an occlusion estimate.
                confidence = float(np.median([o["detector_confidence"] for o in part]))
                score = (-length, -float(np.median(heights)), -confidence, part[0]["frame_index"])
                ranked.append((score, part))
    # Keep diverse alternatives rather than dozens of almost identical windows.
    selected = []
    for score, part in sorted(ranked, key=lambda x: x[0]):
        start, end = part[0]["frame_index"], part[-1]["frame_index"]
        if any(max(0, min(end, x["end"]) - max(start, x["start"]) + 1) /
               min(len(part), x["native_frame_count"]) > .5 for x in selected):
            continue
        selected.append({"start": start, "end": end, "observations": part,
                         "native_frame_count": len(part),
                         "output_slots": (len(part) - 1) * 5 // 2 + 1,
                         "height_median_px": -score[1], "detector_confidence_median": -score[2]})
        if len(selected) == 3:
            break
    return selected, dict(rejected), max(map(len, groups), default=0)


def screen(row, cfg):
    source = ROOT / "predicted_tracks" / row["split"] / row["scene"]
    receipt_path = source / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["source_metadata_sha256"] == sha(Path(row["metadata"]))
    assert receipt["protocol_sha256"] == sha(ROOT / "SCREEN_PROTOCOL.json")
    assert receipt["tracker_sha256"] == sha(ROOT / "bytetrack.yaml")
    assert receipt["worker_sha256"] == sha(ROOT / "track_scene.py")
    candidates, audit = [], []
    for cam in map(int, row["camera_indices"]):
        path = source / f"camera{cam}.json"
        data = json.loads(path.read_text())
        assert data["frame_count"] == row["frames"] and data["camera_index"] == cam
        for track in data["tracks"]:
            windows, rejections, longest = select_windows(track["observations"], cfg)
            state = ("metadata_candidate_not_visibility_accepted" if windows else
                     "short_track" if longest < 25 else "no_metadata_window")
            key = f"{row['segment']}:cam{cam}:{track['track_id']}"
            audit.append({"track_id": track["track_id"], "camera_index": cam,
                          "identity_key": key, "state": state,
                          "longest_contiguous_native_frames": longest,
                          "rejected_window_reasons": rejections})
            if not windows:
                continue
            paths = [Path(row["processed_path"]) / "images" / f"{o['frame_index']:03d}_{cam}.jpg"
                     for window in windows for o in window["observations"]]
            if not all(p.is_file() for p in paths):
                raise FileNotFoundError("Selected source pixels missing")
            candidates.append({"split": row["split"], "scene": row["scene"], "segment": row["segment"],
                               "identity_key": key, "camera_index": cam, "track_id": track["track_id"],
                               "identity_scope": "predicted_camera_local_not_official",
                               "identity_verified": False, "associated_lidar_track_id": None,
                               "windows": windows, "track_cache": str(path), "track_cache_sha256": sha(path),
                               "source_metadata": row["metadata"],
                               "source_metadata_sha256": receipt["source_metadata_sha256"],
                               "camera_timing": "processed_nominal_relative_not_exposure",
                               "occlusion": "unknown", "action": "unknown", "visual_state": "unreviewed",
                               "generation_allowed": False})
    return {"split": row["split"], "scene": row["scene"], "receipt_sha256": sha(receipt_path),
            "screen_sha256": sha(Path(__file__)), "tracks": audit, "candidates": candidates}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch-pid", type=int)
    args = parser.parse_args()
    cfg = json.loads((ROOT / "SCREEN_PROTOCOL.json").read_text())
    assert sha(ROOT / "SCREEN_PROTOCOL.json") == (ROOT / "SCREEN_PROTOCOL.sha256").read_text().strip()
    rows = [r for r in json.loads((ROOT / "inventory.json").read_text())["scenes"]
            if not r["native_labels_tfrecord"]]
    results, errors = {}, {}
    semaphore = asyncio.Semaphore(2)

    async def one(row):
        key = f"{row['split']}/{row['scene']}"
        async with semaphore:
            try:
                result = await asyncio.to_thread(screen, row, cfg)
                write(ROOT / "predicted_scan" / row["split"] / (row["scene"] + ".json"), result)
                results[key] = result
            except Exception as exc:
                errors[key] = repr(exc)

    while True:
        ready = [r for r in rows if f"{r['split']}/{r['scene']}" not in results
                 and f"{r['split']}/{r['scene']}" not in errors
                 and (ROOT / "predicted_tracks" / r["split"] / r["scene"] / "receipt.json").is_file()]
        await asyncio.gather(*(one(r) for r in ready))
        candidates = [c for _, result in sorted(results.items()) for c in result["candidates"]]
        summary = {"updated_at": time.time(), "scene_count": len(results), "target_scene_count": len(rows),
                   "errors": errors, "camera_tracks": sum(len(r["tracks"]) for r in results.values()),
                   "candidate_camera_tracks": len(candidates),
                   "alternative_windows": sum(len(c["windows"]) for c in candidates),
                   "quality_accepted": 0, "verified_unique_people": None,
                   "identity_limit": "Camera-local keys may duplicate people across cameras or fragmented tracks.",
                   "decisions": dict(collections.Counter(t["state"] for r in results.values() for t in r["tracks"]))}
        write(ROOT / "predicted_summary.json", summary)
        write(ROOT / "predicted_candidates.json", {"state": "needs_identity_and_visual_screen", "candidates": candidates})
        if ready:
            print(json.dumps(summary), flush=True)
        if not args.watch_pid or len(results) + len(errors) == len(rows):
            break
        try:
            os.kill(args.watch_pid, 0)
            cmdline = Path(f"/proc/{args.watch_pid}/cmdline").read_bytes()
            assert b"run_tracking.py" in cmdline, "PID reused by another command"
        except (ProcessLookupError, FileNotFoundError, AssertionError):
            print("Tracking process ended; completed receipts screened. Missing scenes remain explicit.", flush=True)
            break
        await asyncio.sleep(30)


if __name__ == "__main__":
    asyncio.run(main())
