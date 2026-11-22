"""Scene-local association proposals only; never merge uncertain person IDs."""
import asyncio
import collections
import json
import time

from associate_identity import associate
from scan_native import ROOT, sha, write


async def main():
    if (ROOT / "predicted_identity_status.json").exists():
        raise RuntimeError("Existing association run requires explicit recovery")
    destination = ROOT / "predicted_identity"
    destination.mkdir(exist_ok=True)
    protocol_path = ROOT / "IDENTITY_PROTOCOL.json"
    expected = (ROOT / "IDENTITY_PROTOCOL.sha256").read_text().strip()
    assert sha(protocol_path) == expected
    algorithm = ROOT / "associate_identity.py"
    algorithm_hash = sha(algorithm)
    protocol = json.loads(protocol_path.read_text())
    grouped = collections.defaultdict(list)
    for candidate in json.loads((ROOT / "predicted_candidates.json").read_text())["candidates"]:
        grouped[candidate["split"], candidate["scene"]].append(candidate)
    inventory = json.loads((ROOT / "inventory.json").read_text())["scenes"]
    scenes = [r for r in inventory if (r["split"], r["scene"]) in grouped]
    slots = asyncio.Semaphore(2)
    results = []
    state = {"started": time.time(), "state": "running", "target_scenes_with_candidates": len(scenes),
             "completed_scenes": 0, "failed_scenes": {}, "automatic_identity_merging": False,
             "algorithm_sha256": algorithm_hash, "protocol_sha256": expected}
    write(ROOT / "predicted_identity_status.json", state)

    async def one(row):
        async with slots:
            key = (row["split"], row["scene"])
            assert sha(protocol_path) == expected and sha(algorithm) == algorithm_hash
            try:
                result = await asyncio.to_thread(associate, row, grouped[key], protocol)
                result["candidate_track_count"] = len(grouped[key])
                result["automatic_identity_merging"] = False
                write(destination / key[0] / (key[1] + ".json"), result)
                results.append(result)
                state["completed_scenes"] += 1
            except Exception as exc:
                state["failed_scenes"]["/".join(key)] = repr(exc)
            write(ROOT / "predicted_identity_status.json", state)

    await asyncio.gather(*(one(row) for row in scenes))
    proposed = []
    conflicts = []
    for result in results:
        by_world = collections.defaultdict(list)
        for link in result["links"]:
            if link["state"] == "proposed_association":
                proposed.append(link)
                by_world[link["proposed_world_track"]].append(link)
        candidate_map = {c["identity_key"]: c for c in grouped[result["split"], result["scene"]]}
        for world, links in by_world.items():
            # Two local tracks in the same camera at overlapping frames cannot
            # silently become a single identity; inspect duplicate detections or mismatch.
            by_camera = collections.defaultdict(list)
            for link in links:
                by_camera[link["camera_index"]].append(link)
            for camera, local_links in by_camera.items():
                occupied = collections.defaultdict(list)
                for link in local_links:
                    candidate = candidate_map[link["identity_key"]]
                    frames = {o["frame_index"] for window in candidate["windows"]
                              for o in window["observations"]}
                    for frame in frames:
                        occupied[frame].append(link["identity_key"])
                overlap = {str(f): keys for f, keys in occupied.items() if len(keys) > 1}
                if overlap:
                    conflicts.append({"split": result["split"], "scene": result["scene"],
                                      "world_track": world, "camera": camera,
                                      "overlap_frames": overlap})
    summary = {
        "updated_at": time.time(), "candidate_camera_tracks": sum(map(len, grouped.values())),
        "examined_scenes": len(results), "scene_states": dict(collections.Counter(r["state"] for r in results)),
        "proposed_links": len(proposed), "same_camera_overlap_conflicts": len(conflicts),
        "conflicts": conflicts, "verified_unique_people": None,
        "automatic_identity_merging": False,
        "limitations": [
            "Prior native validation had12 wrong links among902 proposals; proposals are not ground truth.",
            "Processed scenes use timestamp-union mapping where available; exact raw image exposure is unavailable.",
            "Do not use proposed world-track groups as an accepted unique-person count.",
        ],
    }
    write(ROOT / "predicted_identity_summary.json", summary)
    state.update(state="finished_requires_identity_review", finished=time.time())
    write(ROOT / "predicted_identity_status.json", state)
    print(json.dumps({k: v for k, v in summary.items() if k != "conflicts"}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
