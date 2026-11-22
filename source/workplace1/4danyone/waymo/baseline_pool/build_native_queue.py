"""Prepare full native candidate queue, preserving identity uncertainty."""
import json
from pathlib import Path

from scan_native import ROOT, sha, write


def main():
    output = ROOT / "native_queue.json"
    if output.exists():
        raise RuntimeError("Queue already frozen")
    pilots = {r["identity_key"]: r for r in json.loads((ROOT / "pilot_manifest.json").read_text())["candidates"]}
    candidates = json.loads((ROOT / "native_candidates.json").read_text())["candidates"]
    rows, reused = [], []
    for row in candidates:
        candidate = row.copy()
        if row["identity_key"] in pilots:
            pilot = pilots[row["identity_key"]]
            package = Path(pilot["sample_root"]).parent
            ready = package / "READY.json"
            assert ready.is_file(), "Pilot package must be validated before reuse"
            reused.append({"identity_key": row["identity_key"], "ready": str(ready),
                           "ready_sha256": sha(ready), "quality_accepted": False})
            continue
        import hashlib
        identifier = "T" + hashlib.sha256(row["identity_key"].encode()).hexdigest()[:14]
        rawstat = Path(row["raw_tfrecord"]).stat()
        candidate.update(review_id=identifier, cohort="baseline_pool",
                         raw_stat_at_selection={"size": rawstat.st_size, "mtime_ns": rawstat.st_mtime_ns},
                         track_cache_sha256=sha(Path(row["track_cache"])),
                         sample_root=str(ROOT / "samples" / row["split"] / row["scene"] / identifier / "clip0"))
        rows.append(candidate)
    assert len({r["review_id"] for r in rows}) == len(rows)
    write(output, {"scope": "All native-label metadata candidates; predicted-camera queue separate",
                   "candidates": rows, "reused_pilots": reused,
                   "input_gate_version": 2, "occlusion_alone_not_rejected": True,
                   "generation_allowed": False})
    print(json.dumps({"queued": len(rows), "reused": len(reused)}))


if __name__ == "__main__":
    main()
