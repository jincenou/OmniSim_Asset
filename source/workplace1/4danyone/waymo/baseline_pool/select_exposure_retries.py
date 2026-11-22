"""Find other measured-time windows without relaxing the frozen baseline."""
import asyncio
import collections
import copy
import json
from pathlib import Path
import struct
import sys

import numpy as np

from scan_native import ROOT, load_tracks, sha, write

sys.path.insert(0, "/mnt/ojc/data/waymo_track_bbox_10/python_pkgs")
from waymo_open_dataset import dataset_pb2


def measured_times(row):
    raw = Path(row["native_labels_tfrecord"])
    stat = {"size": raw.stat().st_size, "mtime_ns": raw.stat().st_mtime_ns}
    records = []
    with raw.open("rb") as stream:
        while header := stream.read(12):
            assert len(header) == 12
            length = struct.unpack("<Q", header[:8])[0]
            payload = stream.read(length)
            assert len(payload) == length and len(stream.read(4)) == 4
            frame = dataset_pb2.Frame()
            frame.ParseFromString(payload)
            assert frame.context.name == row["segment"].removeprefix("segment-").removesuffix("_with_camera_labels")
            records.append({
                "frame_timestamp_micros": frame.timestamp_micros,
                "centers": {str(image.name-1): .5*(image.camera_trigger_time+image.camera_readout_done_time)
                            for image in frame.images if image.name in (1, 2, 3)},
            })
    assert len(records) == row["frames"]
    write(ROOT / "research/exposure_times" / row["split"] / (row["scene"]+".json"),
          {"raw_source": str(raw), "raw_stat": stat, "records": records, "crc_checked": False})
    return records


def select(row, failed, cfg):
    times = measured_times(row)
    tracks, cache = load_tracks(row)
    chosen, decisions = [], []
    for key, original in failed:
        options = []
        reasons = collections.Counter()
        relevant = [t for t in tracks if
                    (original.get("associated_lidar_track_id") and
                     t.get("associated_lidar_track_id") == original["associated_lidar_track_id"]) or
                    (t["camera_index"] == original["camera_index"] and t["track_id"] == original["track_id"])]
        for track in relevant:
            obs = sorted(track["observations"], key=lambda o: o["frame_index"])
            for length in cfg["native_lengths"]:
                for start in range(len(obs)-length+1):
                    part = obs[start:start+length]
                    frames = np.array([o["frame_index"] for o in part])
                    if not np.all(np.diff(frames) == 1):
                        continue
                    boxes = np.array([o["bbox_xyxy"] for o in part])
                    height = boxes[:, 3]-boxes[:, 1]
                    if (not np.isfinite(boxes).all() or height.min() < cfg["minimum_height_px"]
                        or np.median(height) < cfg["median_height_px"]
                        or np.any(boxes[:, :2] < 0) or np.any(boxes[:, 2:] > [1920, 1280])):
                        continue
                    for observation in part:
                        assert times[observation["frame_index"]]["frame_timestamp_micros"] == observation["timestamp_micros"]
                    exposure = np.array([times[i]["centers"][str(track["camera_index"])] for i in frames])
                    if not np.all((np.diff(exposure) >= .08) & (np.diff(exposure) <= .12)):
                        reasons["measured_time_gap"] += 1
                        continue
                    deficit = (length-1)/10 - (exposure[-1]-exposure[0])
                    if deficit > .001:
                        reasons["frozen_exposure_coverage"] += 1
                        continue
                    if (track["camera_index"] == original["camera_index"] and
                        frames[0] == original["observations"][0]["frame_index"] and
                        frames[-1] == original["observations"][-1]["frame_index"]):
                        continue
                    difficult = float(np.mean([o["difficulty"] >= 2 for o in part]))
                    score = (length, -difficult, float(np.median(height)), -int(frames[0]))
                    options.append((score, track, part, float(deficit)))
        decision = {"original_review_id": key, "alternative_eligible_windows": len(options),
                    "failed_window_reasons": dict(reasons), "identity_rejected": False}
        if options:
            score, track, part, deficit = max(options, key=lambda option: option[0])
            candidate = copy.deepcopy(original)
            candidate.update(review_id=key+"_time1", camera_index=track["camera_index"],
                             track_id=track["track_id"], observations=part,
                             native_frame_count=len(part), output_slots=(len(part)-1)*5//2+1,
                             height_median_px=score[2],
                             height_min_px=min(o["bbox_xyxy"][3]-o["bbox_xyxy"][1] for o in part),
                             difficulty2_fraction=-score[1], track_cache=str(cache), track_cache_sha256=sha(cache),
                             sample_root=str(ROOT/"retry"/row["split"]/row["scene"]/key/"time1/clip0"),
                             retry_of=original["sample_root"],
                             selection_rule="Same identity, original size/continuity bounds plus actual measured exposure coverage <=1ms deficit; no occlusion exclusion",
                             measured_exposure_deficit_s=deficit)
            chosen.append(candidate)
            decision.update(state="alternative_queued_for_preparation", selected_camera=track["camera_index"],
                            selected_frames=[part[0]["frame_index"], part[-1]["frame_index"]])
        else:
            decision["state"] = "no_alternative_under_current_protocol_not_identity_rejected"
        decisions.append(decision)
    return chosen, decisions


async def main():
    output = ROOT / "retry_time_queue.json"
    if output.exists():
        raise RuntimeError("Existing selection must not be overwritten")
    jobs = json.loads((ROOT/"native_prepare_status.json").read_text())["jobs"]
    grouped = collections.defaultdict(list)
    for key, job in jobs.items():
        if job.get("failed_stage") == "interpolate":
            log = (ROOT/"logs"/f"{key}_interpolate.log").read_text()
            if "Exposure interval lacks coverage" in log:
                candidate = json.loads((Path(job["source"]).parent/"candidate.json").read_text())
                grouped[candidate["split"], candidate["scene"]].append((key, candidate))
    cfg = json.loads((ROOT/"SCREEN_PROTOCOL.json").read_text())
    inventory = json.loads((ROOT/"inventory.json").read_text())["scenes"]
    slots = asyncio.Semaphore(2)
    async def one(row):
        async with slots:
            return await asyncio.to_thread(select, row, grouped[row["split"], row["scene"]], cfg)
    results = await asyncio.gather(*(one(row) for row in inventory if (row["split"], row["scene"]) in grouped))
    candidates = [c for result, _ in results for c in result]
    decisions = [d for _, result in results for d in result]
    write(output, {"candidates": candidates, "generation_allowed": False,
                   "scope": "Retry windows of existing identities; original1ms timing tolerance unchanged"})
    write(ROOT/"research/exposure_retry_selection.json", {"decisions": decisions,
          "retry_count": len(candidates), "selection_sha256": sha(Path(__file__)), "generation_started": False})
    print(json.dumps(decisions), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
