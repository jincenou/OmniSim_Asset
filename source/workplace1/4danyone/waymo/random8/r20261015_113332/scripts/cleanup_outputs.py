"""Remove only this run's redundant previews and completed SAM2 input copies."""
import argparse
import asyncio
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
from zoneinfo import ZoneInfo

from comparison_output import BRANCHES, complete_results, save

R = Path(__file__).resolve().parents[1]
AUDIT = R / "research/output_cleanup.json"
MEDIA = {".mp4", ".jpg", ".jpeg", ".png"}
PREVIOUS = re.compile(r"heldout_view(?:00|08|16)_(?:five|frame\d+_five)_previous\d+\.(?:mp4|jpg|png)")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def active(pid):
    try:
        return str(R).encode() in Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def main():
    prior = json.loads(AUDIT.read_text()) if AUDIT.exists() else {"events": []}
    events = []

    def remove(paths, reason, metadata=None):
        paths = list(paths)
        if not paths:
            return
        assert all(path.is_file() and not path.is_symlink() and R / "samples" in path.parents for path in paths)
        row = {"reason": reason, "files": [str(path.relative_to(R)) for path in paths],
               "file_count": len(paths), "bytes": sum(path.stat().st_size for path in paths)}
        if metadata is not None:
            row["metadata"] = metadata
        # Write the deletion inventory before touching authorized transient files.
        save(R / "research/output_cleanup_pending.json", row)
        for path in paths:
            path.unlink()
        events.append(row)

    for base in sorted((R / "samples").glob("N[0-9][0-9]")):
        if not base.is_dir() or base.is_symlink():
            continue
        comparison = base / "comparison"
        if comparison.exists():
            remove((p for p in comparison.iterdir() if p.is_file() and PREVIOUS.fullmatch(p.name)),
                   "obsolete_previous_comparison_copy")
            results, missing = complete_results(base)
            if missing:
                remove((p for p in comparison.glob("heldout_view*") if p.suffix in MEDIA),
                       "partial_comparison_before_all_four_final_evaluations")
                save(comparison / "status.json", {"state": "waiting_for_all_four_evaluated_branches",
                     "missing_branches": missing, "media_generated": False, "policy": "complete-four-branch-v2"})
            for manifest in sorted((comparison / "parts").glob("*/manifest.json")):
                info = json.loads(manifest.read_text())
                if set(info.get("branches", [])) != set(BRANCHES):
                    remove((p for p in manifest.parent.iterdir() if p.suffix in MEDIA),
                           "partial_body_region_comparison", info)
                    manifest.unlink()
                    manifest.parent.rmdir()
                else:
                    assert results is not None
                    signature = hashlib.sha256("".join(sha(base / "evaluations" / branch / "results.json")
                                                       for branch in BRANCHES).encode()).hexdigest()[:16]
                    assert manifest.parent.name == signature
                    assert all(Path(row["file"]).exists() and sha(Path(row["file"])) == row["sha256"] for row in info["images"])
                    target = comparison / "details"
                    assert not target.exists(), "Inspect existing stable details before merging"
                    manifest.parent.replace(target)
                    for row in info["images"]:
                        row["file"] = str(target / Path(row["file"]).name)
                    info["input_results_signature"] = signature
                    save(target / "manifest.json", info)
                    events.append({"reason": "move_complete_region_previews_to_one_stable_directory",
                                   "from": str(manifest.parent.relative_to(R)), "to": str(target.relative_to(R)), "bytes": 0})
            parts = comparison / "parts"
            if parts.exists() and not any(parts.iterdir()):
                parts.rmdir()

        cache = base / "data/sam2_input"
        receipt = R / "logs" / f"{base.name}_rgba.json"
        proof = R / "research" / f"{base.name}-mask-complete.json"
        if cache.exists() and receipt.exists() and proof.exists():
            job = json.loads(receipt.read_text())
            successful = job.get("exit_code") == 0
            # Adoption can replace the shared receipt with unknown exit status.
            # The same process's independently archived actual exit remains authoritative.
            if not successful and job.get("pid"):
                for archived in (R / "logs/actual_receipts").glob(f"{base.name}_rgba_pid{job['pid']}_exit0*.json"):
                    actual = json.loads(archived.read_text()).get("actual_receipt", {})
                    if actual.get("pid") == job["pid"] and actual.get("exit_code") == 0:
                        successful = True
            if successful and not active(job.get("pid", 0)):
                assert not cache.is_symlink() and all(not p.is_symlink() for p in cache.rglob("*"))
                remove((p for p in cache.rglob("*") if p.is_file()), "completed_disposable_SAM2_JPEG_input_copies")
                shutil.rmtree(cache)

        for directory in sorted((base / "evaluations").glob("*_snapshot*")):
            path = directory / "results.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            assert data.get("snapshot") is not None
            if active(data["device"]["pid"]):
                continue
            media = [p for p in directory.iterdir() if p.suffix in MEDIA]
            if media:
                remove(media, "budget_snapshot_keeps_metrics_and_dynamic_package_without_duplicate_media",
                       {"videos": data.get("videos", {}), "snapshots": data.get("snapshots", [])})
                original_metrics_sha = hashlib.sha256(json.dumps({key: data[key] for key in ("mean", "samples", "dense_anomaly_samples")}, sort_keys=True).encode()).hexdigest()
                data.update(videos={}, snapshots=[], output_retention="budget_metrics_only_v2",
                            retired_preview_count=len(media), metric_payload_sha256_unchanged=original_metrics_sha)
                save(path, data)
        for directory in sorted((base / "evaluations").glob("*_failed_attempt[0-9]*")):
            branch = re.sub(r"_failed_attempt[0-9]+$", "", directory.name)
            if branch in BRANCHES and (base / "evaluations" / branch / "results.json").exists() and not (directory / "results.json").exists():
                remove((p for p in directory.iterdir() if p.suffix in MEDIA), "failed_preview_replaced_by_valid_final_evaluation")
                if not any(directory.iterdir()):
                    directory.rmdir()

    if events:
        prior["events"].extend(events)
        prior.update(updated_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                     scope="Only redundant/partial previews and disposable SAM2 JPEG inputs in this run. Final models, checkpoints, RGBA, original videos, final evaluations, source data and failure logs retained.",
                     deleted_files=sum(row.get("file_count", 0) for row in prior["events"]),
                     released_bytes=sum(row.get("bytes", 0) for row in prior["events"]))
        save(AUDIT, prior)
        (R / "research/output_cleanup_pending.json").unlink(missing_ok=True)
    print(json.dumps({"new_events": len(events), "deleted_files_total": prior.get("deleted_files", 0),
                      "released_bytes_total": prior.get("released_bytes", 0)}), flush=True)


async def watch():
    while True:
        main()
        if json.loads((R / "STATUS.json").read_text())["state"].startswith("all_registered_tasks_terminal"):
            return
        await asyncio.sleep(120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    asyncio.run(watch()) if args.watch else main()
