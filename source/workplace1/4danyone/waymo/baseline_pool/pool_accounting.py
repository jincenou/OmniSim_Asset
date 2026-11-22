"""Keep attempts, windows and verified scene identities in separate denominators."""
import collections
import csv
import io
import json
from pathlib import Path
import time

from package_catalog import ready_paths
from scan_native import ROOT, write


FAILURES = [
    ("empty_mask_retention_check", "np.float64(0.0)"),
    ("camera_radius_limit", "Root-local branch still needs excessive camera radius"),
    ("exposure_coverage", "Exposure interval lacks coverage"),
    ("no_valid_framing", "No valid frames for input framing analysis"),
    ("rigid_geometry_consistency", "Global/incam SMPL geometry is not a consistent rigid pair"),
    ("gpu_memory", "CUDA out of memory"),
    ("crop_exceeds_source", "Required crop exceeds source; no implicit resize"),
]


def summarize(paths=None):
    paths = ready_paths() if paths is None else paths
    packages = []
    for path in paths:
        data = json.loads(path.read_text())
        row = data["candidate"]
        assert data["generation_started"] is False
        review_path = path.parent / "review.json"
        review = json.loads(review_path.read_text()) if review_path.exists() else {}
        packages.append({
            "ready": str(path), "split": row["split"], "scene": row["scene"],
            "identity_key": row["identity_key"], "identity_scope": row["identity_scope"],
            "window_start": row["observations"][0]["frame_index"],
            "window_end": row["observations"][-1]["frame_index"],
            "camera": row["camera_index"], "alternative": path.is_relative_to(ROOT / "retry"),
            "input_review_passed": review.get("input_review_passed") is True,
        })
    ready_directories = {str(path.parent) for path in paths}
    attempts = []
    for name in ["native_prepare_status.json", "predicted_prepare_status.json",
                 "processed_pilot_status.json", "retry_prepare_status.json", "retry2_prepare_status.json", "crop_retry_prepare_status.json", "foreground_retry_prepare_status.json", "alternative_retry_prepare_status.json", "native_mask_retry_prepare_status.json", "terminal_retry_prepare_status.json", "review7_boundary_retry_prepare_status.json", "review7_native_boundary_retry_prepare_status.json", "review8_predicted_retry_prepare_status.json", "review8_native_retry_prepare_status.json", "review9_predicted_retry_prepare_status.json", "review9_native_retry_prepare_status.json"]:
        status_path = ROOT / name
        if not status_path.exists():
            continue
        for key, job in json.loads(status_path.read_text()).get("jobs", {}).items():
            if job["state"] != "preparation_failed_not_identity_rejected":
                continue
            stage = job["failed_stage"]
            log = ROOT / "logs" / f"{key}_{stage}.log"
            text = log.read_text(errors="replace") if log.exists() else ""
            category = next((label for label, signature in FAILURES if signature in text), "unclassified")
            source = job.get("source")
            prepared = bool(source and str(Path(source).parent) in ready_directories)
            attempts.append({
                "review_id": key, "queue": name, "stage": stage, "reason": category,
                "historical_exit": job.get("stages", {}).get(stage, {}).get("exit_code"),
                "subsequently_prepared_same_window": prepared,
                "identity_rejected": False, "source": source, "log": str(log),
            })
    official = {r["identity_key"] for r in packages if r["identity_scope"] == "official_2d3d_association"}
    unresolved = collections.defaultdict(set)
    for row in packages:
        if row["identity_scope"] != "official_2d3d_association":
            unresolved[row["identity_scope"]].add(row["identity_key"])
    windows = {(r["identity_key"], r["camera"], r["window_start"], r["window_end"]) for r in packages}
    summary = {
        "updated_at": time.time(),
        "prepared_packages": len(packages), "distinct_prepared_camera_windows": len(windows),
        "prepared_official_scene_identity_keys": len(official),
        "prepared_unresolved_camera_track_keys_by_scope": {k: len(v) for k, v in unresolved.items()},
        "alternative_window_packages": sum(r["alternative"] for r in packages),
        "input_review_candidate_windows": sum(r["input_review_passed"] for r in packages),
        "input_review_candidate_official_scene_identities": len({
            r["identity_key"] for r in packages
            if r["input_review_passed"] and r["identity_scope"] == "official_2d3d_association"}),
        "quality_accepted_people": None, "all_verified_unique_people": None,
        "historical_failed_attempts": len(attempts),
        "historical_failures_subsequently_prepared_same_window": sum(r["subsequently_prepared_same_window"] for r in attempts),
        "unresolved_failure_categories": dict(collections.Counter(
            r["reason"] for r in attempts if not r["subsequently_prepared_same_window"])),
        "limits": [
            "Official association deduplicates within a scene; it does not establish identity across separate scenes.",
            "Unresolved camera tracks may duplicate each other or official identities; never add them as a people total.",
            "A failed preparation attempt is not an excluded person.",
            "Prepared and no numeric warning do not prove quality acceptance.",
        ],
    }
    write(ROOT / "ACCOUNTING.json", summary)
    write(ROOT / "failure_inventory.json", {"updated_at": summary["updated_at"], "attempts": attempts})
    stream = io.StringIO()
    fields = ["ready", "split", "scene", "identity_key", "identity_scope",
              "window_start", "window_end", "camera", "alternative", "input_review_passed"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(packages)
    temp = ROOT / "prepared_windows.csv.writing"
    temp.write_text(stream.getvalue())
    temp.replace(ROOT / "prepared_windows.csv")
    return summary


if __name__ == "__main__":
    print(json.dumps(summarize(), ensure_ascii=False))
