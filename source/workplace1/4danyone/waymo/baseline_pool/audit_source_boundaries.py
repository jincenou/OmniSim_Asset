"""Source annotation boundary diagnostic; never a quality or exclusion gate."""
import csv
import io
import json
from pathlib import Path
import time

from PIL import Image

from package_catalog import ready_paths
from scan_native import ROOT, write


def main():
    rows = []
    for path in ready_paths():
        candidate = json.loads(path.read_text())["candidate"]
        native = json.loads((path.parent / "clip0/native.json").read_text())
        # Coordinates are in the original distorted image, not the crop canvas.
        with Image.open(native["frames"][0]["raw_source_path"]) as image:
            width, height = image.size
        assert (width, height) == (1920, 1280), "Unsupported original image dimensions"
        flags = []
        for observation in candidate["observations"]:
            x0, y0, x1, y1 = observation["bbox_xyxy"]
            flags.append(x0 <= .5 or y0 <= .5 or x1 >= width-.5 or y1 >= height-.5)
        run, longest = 0, 0
        for value in flags:
            run = run+1 if value else 0
            longest = max(longest, run)
        rows.append({
            "split": candidate["split"], "scene": candidate["scene"],
            "review_id": candidate["review_id"], "package": str(path.parent),
            "camera": candidate["camera_index"], "native_frames": len(flags),
            "boundary_touch_frames": sum(flags),
            "boundary_touch_fraction": sum(flags)/len(flags),
            "longest_consecutive_native_frames": longest,
            "first_frame_touches": flags[0], "last_frame_touches": flags[-1],
            "quality_decision": "not_evaluated_by_this_diagnostic",
        })
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    temp = ROOT / "source_boundary_diagnostic.csv.writing"
    temp.write_text(stream.getvalue())
    temp.replace(ROOT / "source_boundary_diagnostic.csv")
    summary = {
        "updated_at": time.time(), "prepared_windows": len(rows),
        "any_boundary_touch_windows": sum(r["boundary_touch_frames"] > 0 for r in rows),
        "last_frame_touch_windows": sum(r["last_frame_touches"] for r in rows),
        "longest_run_at_least_3_native_frames": sum(r["longest_consecutive_native_frames"] >= 3 for r in rows),
        "diagnostic_definition": "Original distorted bbox edge within0.5px of original1920x1280 image border",
        "counts_are_windows_not_unique_people": True,
        "auto_exclusion": False,
        "limits": [
            "Boundary contact alone does not prove body truncation, bad skeleton or generation failure.",
            "Annotation/detector boxes are not visible-body masks; no true occlusion ratio is inferred.",
            "The diagnostic does not change frozen selection, source inputs or generation227.",
        ],
    }
    write(ROOT / "source_boundary_diagnostic.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
