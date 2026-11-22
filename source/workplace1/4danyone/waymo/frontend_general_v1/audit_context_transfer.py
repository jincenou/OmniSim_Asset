"""Compare original/repaired mask support without changing motion conditions."""
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

from audit_context_conditioning import spans
from optimize_motion import supported_points

G = Path(__file__).resolve().parent
B = G / "text_motion/bilateral_context"
OUT = B / "transfer_check"
SOURCE = B / "P18/calibrated/input"


def main():
    output = OUT / "support_comparison.json"
    assert not output.exists()
    state = load_file(str(SOURCE / "motion_pair/calibrated_moving/motion.safetensors"))
    kp = state["observed_keypoints_2d"].numpy()
    cfg = json.loads((G / "protocol_v1.json").read_text())["visibility"]
    result = {"scope": "Unchanged detector, fixed original prompts and visibility thresholds; only SAM transfer synchronization differs.",
              "quality_approved": False, "does_not_establish_true_limb_visibility": True,
              "source_detector_sha256": hashlib.sha256((SOURCE / "motion_pair/calibrated_moving/motion.safetensors").read_bytes()).hexdigest(),
              "arms": {}}
    for arm, directory in (("box", "context_segmentation"), ("points", "context_segmentation_points")):
        old_path = SOURCE / directory / "target_masks.npz"
        new_path = OUT / "P18" / arm / "target_masks.npz"
        old = np.load(old_path)["masks"]
        new = np.load(new_path)["masks"]
        assert old.shape == new.shape and len(old) == len(kp) == 101
        a, b = old > 127, new > 127
        overlap = (a & b).sum((1, 2)) / np.maximum((a | b).sum((1, 2)), 1)
        old_support, _ = supported_points(kp, old, cfg)
        new_support, _ = supported_points(kp, new, cfg)
        row = {
            "old_sha256": hashlib.sha256(old_path.read_bytes()).hexdigest(),
            "new_sha256": hashlib.sha256(new_path.read_bytes()).hexdigest(),
            "pixel_max_change_255": int(np.abs(old.astype(np.int16) - new.astype(np.int16)).max()),
            "pixel_changed_count": int((old != new).sum()),
            "binary_iou_min_median": [float(overlap.min()), float(np.median(overlap))],
            "old_empty_frames": np.flatnonzero(a.sum((1, 2)) < 100).tolist(),
            "new_empty_frames": np.flatnonzero(b.sum((1, 2)) < 100).tolist(),
            "joint_support_changed": int((old_support != new_support).sum()),
            "gained_joint_support": int((~old_support & new_support).sum()),
            "lost_joint_support": int((old_support & ~new_support).sum()),
            "joints": {},
        }
        for name, index in (("left_knee", 13), ("right_knee", 14), ("left_ankle", 15), ("right_ankle", 16)):
            row["joints"][name] = {
                "old_runs": spans(old_support[:, index], np.arange(101) / 25),
                "new_runs": spans(new_support[:, index], np.arange(101) / 25),
                "changed_slots": np.flatnonzero(old_support[:, index] != new_support[:, index]).tolist(),
            }
        result["arms"][arm] = row
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({arm: {k: v for k, v in row.items() if k != "joints"} for arm, row in result["arms"].items()}))


if __name__ == "__main__":
    main()
