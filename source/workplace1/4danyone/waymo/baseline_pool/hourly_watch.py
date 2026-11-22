"""Hourly technical checks without model/API calls or automatic job mutation."""
import asyncio
from collections import Counter
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo

import av
import numpy as np

ROOT = Path(__file__).resolve().parent
GEN = ROOT / "generation227"
OUT = ROOT / "monitor"


def read(path):
    return json.loads(path.read_text())


def save(path, payload):
    temp = path.with_suffix(".writing")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def check_output(row):
    c = row["candidate"]
    base = GEN / "scenes" / c["split"] / c["scene"] / c["review_id"]
    result = {"id": row["id"], "errors": [], "warnings": []}
    try:
        manifest = read(base / "manifest.json")
        receipt = read(base / "validation.json")
        assert manifest["views"] == len(receipt["videos"]) == 24
        assert manifest["frames"] == c["output_slots"]
        for video in receipt["videos"]:
            path = base / "videos" / f"{video['view']:02d}.mp4"
            assert path.is_file(), str(path)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == video["sha256"], str(path)
        assert (base / "contact.jpg").is_file()
        checked = []
        with av.open(str(base / "preview.mp4")) as container:
            container.streams.video[0].thread_count = 1
            count = 0
            for frame in container.decode(video=0):
                assert (frame.width, frame.height) == (792, 1000)
                if count in (0, c["output_slots"]//2, c["output_slots"]-1):
                    rgb = frame.to_ndarray(format="rgb24")
                    checked.append({"frame": count, "mean": float(rgb.mean()), "std": float(rgb.std())})
                count += 1
        assert count == c["output_slots"]
        if any(x["std"] < 2 or x["mean"] < 2 or x["mean"] > 253 for x in checked):
            result["warnings"].append("Extreme preview pixel statistics; inspect image, not an automatic rejection")
        result.update(preview_frames=count, sampled_pixel_statistics=checked, original_video_hashes_checked=24)
    except Exception as error:
        result["errors"].append(repr(error))
    result["quality_passed"] = None
    return result


def inspect():
    rows = {r["id"]: r for r in read(GEN / "manifest.json")["rows"]}
    state = read(GEN / "STATUS.json")
    previous = read(OUT / "latest.json") if (OUT / "latest.json").exists() else {}
    checks = dict(previous.get("output_checks", {}))
    alerts, live = [], []
    for key, job in state["jobs"].items():
        if job["state"] == "complete" and (key not in checks or checks[key]["errors"]):
            checks[key] = check_output(rows[key])
        if job["state"] in ("generating", "validating"):
            pid = job["pid"]
            proc = Path("/proc") / str(pid) / "cmdline"
            args = proc.read_bytes().split(b"\0") if proc.exists() else []
            alive = any(a.endswith(b"/generate_single.py") or a.endswith(b"/deliver.py") for a in args)
            record = {"id": key, "pid": pid, "state": job["state"], "process_live": alive}
            log = GEN / "logs" / f"{key}_{job['state']}.log"
            age = time.time()-log.stat().st_mtime if log.exists() else None
            record["seconds_since_log_write"] = age
            live.append(record)
            if not alive:
                alerts.append(f"{key}: recorded process absent; recheck before any recovery")
            elif age is not None and age > 3600:
                alerts.append(f"{key}: no log write for1h; possible stall, not confirmed")
        elif job["state"] == "failed":
            alerts.append(f"{key}: failed; see generation logs")
    # Re-read live state after output checks so concurrent completion isn't
    # interpreted as a missing process from an old status snapshot.
    refreshed = read(GEN / "STATUS.json")
    for record in live:
        if refreshed["jobs"].get(record["id"], {}).get("state") != record["state"]:
            alerts = [a for a in alerts if not a.startswith(record["id"] + ":")]
            record["state_changed_during_check"] = True
    free = shutil.disk_usage(ROOT).free / 2**30
    if free < 12:
        alerts.append("Disk below12GiB; admission may wait. No data automatically removed.")
    for key, check in checks.items():
        alerts.extend(f"{key}: {e}" for e in check["errors"] + check["warnings"])
    preparation = {}
    for name in ("native_prepare_status.json", "predicted_prepare_status.json"):
        prep = read(ROOT / name)
        preparation[name] = {"state": prep["state"], "counts": dict(Counter(j["state"] for j in prep["jobs"].values()))}
    result = {
        "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "next_check_unix": time.time()+3600, "interval_seconds": 3600,
        "generation_state": refreshed["state"],
        "generation_counts": dict(Counter(j["state"] for j in refreshed["jobs"].values())),
        "preparation": preparation, "free_disk_gib": free, "active_processes": live,
        "output_checks": checks, "alerts": alerts,
        "limits": "Technical checks only. Natural motion, identity, hidden limbs, dark clothing and temporal artifacts need visual review. No automatic restart, pruning, rejection, model/API calls or external notifications.",
    }
    save(OUT / "latest.json", result)
    with (OUT / "history.jsonl").open("a") as stream:
        stream.write(json.dumps({k: v for k, v in result.items() if k != "output_checks"}, ensure_ascii=False) + "\n")
    terminal = refreshed["state"] != "running" and all(v["state"] != "running" for v in preparation.values())
    return terminal


async def main():
    OUT.mkdir(exist_ok=True)
    lock = (OUT / "watch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        try:
            finished = await asyncio.to_thread(inspect)
        except Exception as error:
            save(OUT / "watch_error.json", {"at": time.time(), "error": repr(error), "retry_in_seconds": 3600})
            finished = False
        if finished:
            break
        # The detached background process sleeps; no model polling loop is used.
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
