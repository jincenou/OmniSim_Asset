"""Auxiliary world-track association, never replaces baseline GVHMR motion.

Validate on existing official 2D/3D associations before using predicted links.
No face recognition; only within-scene track identifiers and camera geometry.
"""
import asyncio
import json
from pathlib import Path
import time

import cv2
import numpy as np

from scan_native import ROOT, write, sha

DATA = Path("/mnt/public_data/Waymo-3DSkelMo/waymo_open_dataset_3dskelmo/10fps")


def associate(row, candidates, protocol):
    segment = row["segment"].removeprefix("segment-").removesuffix("_with_camera_labels")
    directory = DATA / ("training" if row["split"] == "train" else "validation") / segment
    if not directory.is_dir():
        return {"split": row["split"], "scene": row["scene"], "state": "no_auxiliary_tracks", "links": []}
    identities, translations, source_hashes, available = [], [], {}, []
    raw_tracks = []
    skipped = {}
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            trans, times = data["trans"].astype(np.float64), data["waymo_timestamps"]
        if trans.shape != (len(times), 3) or times.ndim != 1 or len(times) == 0:
            skipped[path.stem] = "invalid_arrays"
            continue
        if not np.isfinite(trans).all() or not np.all(np.diff(times) > 0):
            skipped[path.stem] = "invalid_position_or_time"
            continue
        raw_tracks.append((path, trans, times))
    canonical = np.full(row["frames"], -1, dtype=np.int64)
    for candidate in candidates:
        observations = candidate.get("observations", candidate.get("windows", [{}])[0].get("observations"))
        for observation in observations:
            if "timestamp_micros" in observation:
                index, timestamp = observation["frame_index"], observation["timestamp_micros"]
                assert canonical[index] in (-1, timestamp)
                canonical[index] = timestamp
    if not np.any(canonical >= 0):
        if not raw_tracks:
            return {"split": row["split"], "scene": row["scene"], "state": "no_auxiliary_tracks", "links": []}
        union = np.unique(np.concatenate([x[2] for x in raw_tracks]))
        if len(union) != row["frames"] or not np.all((np.diff(union)/1e6 >= .08) & (np.diff(union)/1e6 <= .12)):
            return {"split": row["split"], "scene": row["scene"], "state": "unknown_full_scene_time_mapping", "links": []}
        canonical = union
    time_to_frame = {int(t): i for i, t in enumerate(canonical) if t >= 0}
    for path, trans, times in raw_tracks:
        aligned = np.zeros((row["frames"], 3), dtype=np.float64)
        observed = np.zeros(row["frames"], dtype=bool)
        for position, timestamp in zip(trans, times, strict=True):
            index = time_to_frame.get(int(timestamp))
            if index is not None:
                aligned[index] = position
                observed[index] = True
        if not observed.any():
            skipped[path.stem] = "no_exact_timestamp_overlap"
            continue
        identities.append(path.stem)
        translations.append(aligned)
        available.append(observed)
        source_hashes[path.stem] = sha(path)
    if not identities:
        return {"split": row["split"], "scene": row["scene"], "state": "no_compatible_auxiliary_tracks",
                "links": [], "skipped": skipped}
    xyz_world = np.asarray(translations)
    scene = Path(row["processed_path"])
    axes = np.eye(4)
    axes[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    projected = {}
    cv2.setNumThreads(1)
    for camera in sorted({c["camera_index"] for c in candidates}):
        intrinsic = np.loadtxt(scene / "intrinsics" / f"{camera}.txt").reshape(-1)
        K = np.array([[intrinsic[0], 0, intrinsic[2]], [0, intrinsic[1], intrinsic[3]], [0, 0, 1]])
        camera_to_world = np.stack([np.loadtxt(scene / "cam_to_world" / f"{i:03d}_{camera}.txt") @ axes
                                    for i in range(row["frames"])])
        w2c = np.linalg.inv(camera_to_world)
        xyz = np.einsum("fij,nfj->nfi", w2c[:, :3, :3], xyz_world) + w2c[None, :, :3, 3]
        pixels, _ = cv2.projectPoints(xyz.reshape(-1, 3), np.zeros(3), np.zeros(3), K, intrinsic[4:])
        pixels = pixels.reshape(len(identities), row["frames"], 2)
        valid = np.asarray(available) & (xyz[..., 2] > .1) & np.isfinite(pixels).all(-1) & (np.linalg.norm(xyz_world, axis=-1) > 1e-6)
        projected[camera] = pixels, valid
    links = []
    for candidate in candidates:
        observations = candidate.get("observations", candidate.get("windows", [{}])[0].get("observations"))
        frames = np.array([o["frame_index"] for o in observations], dtype=int)
        boxes = np.array([o["bbox_xyxy"] for o in observations])
        official = candidate.get("associated_lidar_track_id")
        if "timestamp_micros" in observations[0] and not np.array_equal(
                [o["timestamp_micros"] for o in observations], canonical[frames]):
            links.append({"identity_key": candidate["identity_key"], "state": "timestamp_mismatch",
                          "official": official})
            continue
        pixels, valid = projected[candidate["camera_index"]]
        q, ok = pixels[:, frames], valid[:, frames]
        wh = boxes[:, 2:] - boxes[:, :2]
        center = boxes[:, :2] + wh * [0.5, protocol["vertical_anchor_fraction"]]
        distance = np.linalg.norm((q-center[None])/np.maximum(wh[None], 1), axis=-1)
        inside = ok & (q >= boxes[None, :, :2]).all(-1) & (q <= boxes[None, :, 2:]).all(-1)
        distance[~ok] = 1e6
        scores = np.median(distance, axis=1)
        order = np.argsort(scores, kind="stable")
        best = int(order[0])
        second = float(scores[order[1]]) if len(order) > 1 else None
        margin = second - float(scores[best]) if second is not None else None
        accepted = bool(inside[best].mean() >= protocol["minimum_inside_fraction"] and
                        scores[best] <= protocol["maximum_median_normalized_distance"] and
                        (margin is None or margin >= protocol["minimum_runner_up_margin"]) and
                        ok[best].sum() >= protocol["minimum_valid_frames"])
        links.append({"identity_key": candidate["identity_key"], "track_id": candidate["track_id"],
                      "camera_index": candidate["camera_index"], "official": official,
                      "state": "proposed_association" if accepted else "ambiguous",
                      "proposed_world_track": identities[best] if accepted else None,
                      "best_world_track": identities[best], "inside_fraction": float(inside[best].mean()),
                      "median_normalized_distance": float(scores[best]), "runner_up_margin": margin,
                      "official_match": identities[best] == official if accepted and official else None,
                      "time_basis": "exact_raw_timestamp_checked" if "timestamp_micros" in observations[0]
                                    else "scene_name_and_full_timestamp_union_count; raw_time_unavailable"})
    return {"split": row["split"], "scene": row["scene"], "state": "association_diagnostic_only",
            "links": links, "auxiliary_track_sha256": source_hashes, "skipped": skipped,
            "algorithm_sha256": sha(Path(__file__)), "motion_replaced": False}


async def main():
    protocol_path = ROOT / "IDENTITY_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text())
    assert sha(protocol_path) == (ROOT / "IDENTITY_PROTOCOL.sha256").read_text().strip()
    inventory = json.loads((ROOT / "inventory.json").read_text())["scenes"]
    # Official associations are evaluation labels, never used in the ranking.
    rows = [r for r in inventory if r["native_labels_tfrecord"]]
    semaphore = asyncio.Semaphore(2)
    async def one(row):
        async with semaphore:
            native = json.loads((ROOT / "native_scan" / row["split"] / (row["scene"]+".json")).read_text())
            result = await asyncio.to_thread(associate, row, native["candidates"], protocol)
            write(ROOT / "identity_validation" / row["split"] / (row["scene"]+".json"), result)
            return result
    results = await asyncio.gather(*(one(r) for r in rows))
    links = [link for r in results for link in r["links"] if link.get("official")]
    accepted = [x for x in links if x["state"] == "proposed_association"]
    wrong = [x for x in accepted if x["official_match"] is False]
    summary = {"updated_at": time.time(), "scenes_examined": len(results), "official_camera_candidates": len(links),
               "proposed_links": len(accepted), "wrong_links": len(wrong),
               "precision_on_proposed": (len(accepted)-len(wrong))/len(accepted) if accepted else None,
               "wrong_examples": wrong, "not_automatic_ground_truth": True,
               "predicted_queue_association_started": False, "protocol_sha256": sha(protocol_path)}
    write(ROOT / "identity_validation_summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
