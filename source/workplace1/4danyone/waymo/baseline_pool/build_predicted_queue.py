"""Queue predicted camera tracks; final person counts require identity grouping."""
import hashlib
import json
from pathlib import Path

from scan_native import ROOT, sha, write


def main():
    output = ROOT / "predicted_queue.json"
    if output.exists():
        raise RuntimeError("Predicted queue already frozen")
    pilots = {r["identity_key"]: r
              for r in json.loads((ROOT / "processed_pilot_queue.json").read_text())["candidates"]}
    candidates = json.loads((ROOT / "predicted_candidates.json").read_text())["candidates"]
    rows, reused = [], []
    for row in candidates:
        if row["identity_key"] in pilots:
            ready = Path(pilots[row["identity_key"]]["sample_root"]).parent / "READY.json"
            assert ready.exists()
            reused.append({"identity_key": row["identity_key"], "ready": str(ready), "ready_sha256": sha(ready)})
            continue
        candidate = row.copy()
        windows = candidate.pop("windows")
        candidate.update(windows[0])
        identifier = "D" + hashlib.sha256(row["identity_key"].encode()).hexdigest()[:14]
        candidate.update(review_id=identifier, cohort="baseline_pool_processed_camera",
                         sample_root=str(ROOT / "samples" / row["split"] / row["scene"] / identifier / "clip0"),
                         alternative_frame_windows=[{"start": x["start"], "end": x["end"],
                                                     "native_frame_count": x["native_frame_count"]} for x in windows[1:]],
                         person_count_status="camera_track_only_identity_grouping_pending")
        rows.append(candidate)
    assert len({r["review_id"] for r in rows}) == len(rows)
    write(output, {"scope": "All predicted-camera candidates; camera tracks are NOT distinct person counts",
                   "candidates": rows, "reused_pilots": reused, "input_gate_version": 2,
                   "generation_allowed": False,
                   "deduplication": "Retain provenance for grouping; do not add these counts to official unique identities."})
    print(json.dumps({"queued_camera_tracks": len(rows), "reused": len(reused)}))


if __name__ == "__main__":
    main()
