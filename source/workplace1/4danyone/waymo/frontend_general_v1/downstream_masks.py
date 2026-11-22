"""Execute the frozen existing mask worker, then remove only its JPEG scratch."""
import json
import os
from pathlib import Path
import shutil
import sys

G = Path(__file__).resolve().parent
D = G / "validation12/downstream"
sys.path.insert(0, str(D / "scripts"))
import full_masks


def main():
    assert len(full_masks.GPU_IDS) == 1
    assert int(os.environ["CUDA_VISIBLE_DEVICES"]) == full_masks.GPU_IDS[0]
    full_masks.worker(0)
    report = D / "research" / f"{full_masks.PREFIX}-rank0.json"
    records = json.loads(report.read_text())
    assert [r["view"] for r in records] == list(range(24))
    assert all(r["frames"] == full_masks.FRAMES for r in records)
    # The original source videos are retained; these JPEGs are exclusively worker scratch.
    scratch = full_masks.DEST / "sam2_input"
    assert scratch.is_relative_to(D / "data") and not scratch.is_symlink()
    shutil.rmtree(scratch)
    (D / "research" / f"{full_masks.PREFIX}-complete.json").write_text(json.dumps({
        "state": "24_view_RGBA_complete_not_quality_approved",
        "source": str(full_masks.SOURCE), "frames": full_masks.FRAMES, "views": 24,
        "dimensions": [352, 640], "quality_approved": False,
        "warnings": [r["view"] for r in records if r["quality_warning"]],
        "alpha": "Original SAM2 tracking seeded with per-view BiRefNet, checked every15 frames.",
        "scratch_JPEG_removed_after_all_views_completed": True,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
