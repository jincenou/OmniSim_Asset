"""Reuse calibrated source extraction for a full same-camera identity track."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["camera", "crop"], required=True)
    args = parser.parse_args()
    original = json.loads((args.source / "candidate.json").read_text())
    cache = json.loads(Path(original["cache"]).read_text())
    matches = [t for t in cache["tracks"]
               if t["camera_index"] == original["camera_index"]
               and t["track_id"] == original["track_id"]
               and t["associated_lidar_track_id"] == original["associated_lidar_track_id"]]
    if len(matches) != 1:
        raise ValueError("Identity association must be unique")
    observations = matches[0]["observations"]
    # These fields are inputs to the existing read-only camera audit, not a
    # replacement selection manifest or a new production output window.
    row = {key: original[key] for key in (
        "review_id", "cohort", "split", "scene", "segment", "camera_index",
        "track_id", "raw_tfrecord", "cache", "associated_lidar_track_id")}
    row.update(observations=observations, native_frame_count=len(observations))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "source_manifest.json"
    content = json.dumps({
        "scope": "Source-only extended context; original output time window unchanged",
        "original_candidate": str(args.source / "candidate.json"),
        "candidates": [row],
    }, indent=2) + "\n"
    if manifest.exists():
        if manifest.read_text() != content:
            raise ValueError("Context source manifest changed")
    else:
        manifest.write_text(content)
    scripts = Path(__file__).resolve().parent.parent / "scene20/r20261019/scripts"
    if args.stage == "camera":
        audit = load("context_camera_audit", scripts / "raw_camera_audit.py")
        audit.R = args.output
        sys.argv = ["raw_camera_audit.py", "--review-id", row["review_id"], "--manifest", str(manifest)]
        audit.main()
        return
    os.environ["FDANYONE_SAMPLE_ID"] = row["review_id"]
    os.environ["FDANYONE_SAMPLE_ROOT"] = str(args.output / "input")
    prepare = load("context_crop", scripts / "m02_prepare.py")
    prepare.R = args.output
    prepare.main()
    (args.output / "status.json").write_text(json.dumps({
        "state": "calibrated_context_prepared_no_motion_inference",
        "native_frames": len(observations),
        "first_frame": observations[0]["frame_index"],
        "last_frame": observations[-1]["frame_index"],
        "new_generation": False,
        "quality_approved": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
