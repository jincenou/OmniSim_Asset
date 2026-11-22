"""Read-only native-label screening. No motion fitting or image generation."""
import asyncio
import collections
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
WAYMO = ROOT.parent
PUBLIC = Path("/mnt/public_data/waymo")
EXPORT = Path("/mnt/ojc/data/waymo_track_bbox_10/exported")
CAMERAS = {"FRONT": 0, "FRONT_LEFT": 1, "FRONT_RIGHT": 2}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".writing")
    temp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def load_tracks(row):
    raw = Path(row["native_labels_tfrecord"])
    if row["split"] == "val":
        path = WAYMO / "reconstruction/track_reserve/native" / (row["scene"] + ".json")
        data = json.loads(path.read_text())
        assert data["segment"] == row["segment"]
        stat = raw.stat()
        assert data["raw_stat"] == {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        assert len(data["poses"]) == row["frames"]
        np.testing.assert_allclose(data["poses"][0], np.loadtxt(
            Path(row["processed_path"]) / "ego_to_world/000.txt"), atol=1e-5, rtol=0)
        return data["tracks"], path
    path = EXPORT / row["scene"] / "camera_tracks.json"
    summaries = json.loads((EXPORT / "summary.json").read_text())
    summary = next(x for x in summaries if x["scene_id"] == row["scene"])
    assert summary["context_name"] == row["segment"].removeprefix("segment-").removesuffix("_with_camera_labels")
    assert summary["frame_count"] == row["frames"] and summary["frame_count_matches_public"]
    tracks = []
    for track in json.loads(path.read_text()):
        if track["class_name"] != "pedestrian" or track["camera_name"] not in CAMERAS:
            continue
        observations = []
        for obs in track["observations"]:
            box = obs["bbox_2d"]
            observations.append(dict(frame_index=obs["frame_index"],
                timestamp_micros=obs["timestamp_micros"],
                bbox_xyxy=[box[k] for k in ("xmin", "ymin", "xmax", "ymax")],
                difficulty=obs["detection_difficulty"]))
        tracks.append(dict(camera_index=CAMERAS[track["camera_name"]],
            track_id=track["track_id"],
            associated_lidar_track_id=track["associated_lidar_track_id"],
            observations=observations))
    return tracks, path


def screen(row, cfg):
    tracks, source = load_tracks(row)
    candidates, audit = [], []
    metadata = json.loads(Path(row["metadata"]).read_text())
    for track in tracks:
        obs = sorted(track["observations"], key=lambda x: x["frame_index"])
        frames = np.array([o["frame_index"] for o in obs])
        assert len(set(frames)) == len(frames)
        groups = np.split(np.arange(len(obs)), np.flatnonzero(np.diff(frames) != 1) + 1)
        lengths = [len(g) for g in groups]
        options = []
        camera = track["camera_index"]
        identity = row["segment"] + ":" + (track["associated_lidar_track_id"] or
                                           f"cam{camera}:{track['track_id']}")
        rejects = collections.Counter()
        for length in cfg["native_lengths"]:
            for group in groups:
                for offset in range(max(0, len(group) - length + 1)):
                    part = [obs[int(i)] for i in group[offset:offset + length]]
                    boxes = np.array([o["bbox_xyxy"] for o in part])
                    h = boxes[:, 3] - boxes[:, 1]
                    if not np.isfinite(boxes).all() or np.any(h <= 0):
                        rejects["invalid_box"] += 1
                        continue
                    if h.min() < cfg["minimum_height_px"] or np.median(h) < cfg["median_height_px"]:
                        rejects["resolution"] += 1
                        continue
                    # Baseline camera0..2 images are 1920x1280; tolerate near-edge
                    # boxes but not extrapolated boxes beyond the observed image.
                    if np.any(boxes[:, :2] < 0) or np.any(boxes[:, 2:] > [1920, 1280]):
                        rejects["outside_image"] += 1
                        continue
                    times = np.array([o["timestamp_micros"] for o in part], dtype=np.int64)
                    dt = np.diff(times) / 1e6
                    if np.any((dt < .08) | (dt > .12)):
                        rejects["timestamp_gap"] += 1
                        continue
                    difficult = float(np.mean([o["difficulty"] >= 2 for o in part]))
                    score = (-difficult, float(np.median(h)), float(h.min()), -part[0]["frame_index"])
                    options.append((score, part, length))
            if options:
                break
        decision = {"identity_key": identity, "camera_index": camera,
                    "track_id": track["track_id"], "observations": len(obs),
                    "longest_contiguous_native_frames": max(lengths, default=0),
                    "rejected_window_reasons": dict(rejects)}
        if not options:
            decision["state"] = "short_track" if max(lengths, default=0) < 25 else "no_metadata_window"
            audit.append(decision)
            continue
        score, part, length = max(options, key=lambda x: x[0])
        if not all((Path(row["processed_path"]) / "images" /
                    f"{o['frame_index']:03d}_{camera}.jpg").is_file() for o in part):
            decision["state"] = "missing_selected_pixels"
            audit.append(decision)
            continue
        decision["state"] = "metadata_candidate_not_visibility_accepted"
        audit.append(decision)
        candidates.append(dict(split=row["split"], scene=row["scene"],
            segment=row["segment"], camera_index=camera, track_id=track["track_id"],
            associated_lidar_track_id=track["associated_lidar_track_id"],
            identity_key=identity, identity_scope="official_2d3d_association" if
            track["associated_lidar_track_id"] else "official_camera_local_only",
            observations=part, native_frame_count=length, output_slots=(length - 1) * 5 // 2 + 1,
            height_median_px=score[1], height_min_px=score[2], difficulty2_fraction=-score[0],
            raw_tfrecord=row["native_labels_tfrecord"], track_cache=str(source),
            source_metadata=row["metadata"], visual_state="unreviewed",
            action="unknown", occlusion="unknown", generation_allowed=False))
    return dict(split=row["split"], scene=row["scene"], source=str(source),
                source_sha256=sha(source), tracks=audit, candidates=candidates,
                camera_track_count=len(tracks), metadata_fps=metadata["fps"])


async def main():
    cfg_path = ROOT / "SCREEN_PROTOCOL.json"
    cfg = json.loads(cfg_path.read_text())
    assert sha(cfg_path) == (ROOT / "SCREEN_PROTOCOL.sha256").read_text().strip()
    rows = [r for r in json.loads((ROOT / "inventory.json").read_text())["scenes"]
            if r["native_labels_tfrecord"]]
    semaphore = asyncio.Semaphore(2)
    async def one(row):
        async with semaphore:
            result = await asyncio.to_thread(screen, row, cfg)
            write(ROOT / "native_scan" / row["split"] / (row["scene"] + ".json"), result)
            return result
    results = await asyncio.gather(*(one(row) for row in rows), return_exceptions=True)
    errors = {f"{r['split']}/{r['scene']}": repr(v) for r, v in zip(rows, results)
              if isinstance(v, BaseException)}
    good = [v for v in results if not isinstance(v, BaseException)]
    options = [c for v in good for c in v["candidates"]]
    groups = collections.defaultdict(list)
    for candidate in options:
        groups[candidate["identity_key"]].append(candidate)
    selected = [min(v, key=lambda c: (-c["native_frame_count"], c["difficulty2_fraction"],
                                     -c["height_median_px"], c["camera_index"]))
                for _, v in sorted(groups.items())]
    write(ROOT / "native_candidates.json", {"state": "metadata_candidates_need_visual_screen",
          "protocol_sha256": sha(cfg_path), "candidates": selected,
          "identity_limit": "Camera-local tracks without official association may still duplicate physical people."})
    write(ROOT / "native_summary.json", {"updated_at": time.time(), "scene_count": len(good),
          "errors": errors, "camera_tracks": sum(v["camera_track_count"] for v in good),
          "metadata_camera_candidates": len(options), "deduplicated_candidate_keys": len(selected),
          "official_associated_candidates": sum(c["identity_scope"] == "official_2d3d_association" for c in selected),
          "camera_local_candidates": sum(c["identity_scope"] == "official_camera_local_only" for c in selected),
          "quality_accepted": 0, "unscanned_scenes": 1000 - len(good),
          "decisions": dict(collections.Counter(t["state"] for v in good for t in v["tracks"]))})
    print((ROOT / "native_summary.json").read_text(), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
