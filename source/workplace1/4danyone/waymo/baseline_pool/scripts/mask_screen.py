"""Mask evidence risks; partial missing limbs must reach GVHMR evaluation."""
import json
import hashlib
from pathlib import Path
import numpy as np


def evaluate(masks, boxes, protocol):
    binary = masks > 127
    box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    assert np.all(box_area > 0)
    area = binary.sum((1, 2)) / box_area
    cores = []
    for mask, b in zip(binary, boxes, strict=True):
        x0, y0, x1, y1 = b
        w, h = x1-x0, y1-y0
        xa, ya, xb, yb = np.rint([x0+.3*w, y0+.2*h, x0+.7*w, y0+.6*h]).astype(int)
        patch = mask[max(ya, 0):min(yb, mask.shape[0]), max(xa, 0):min(xb, mask.shape[1])]
        cores.append(float(patch.mean()) if patch.size else 0.)
    core = np.array(cores)
    bad = (area < protocol["low_mask_bbox_area"]) | (core < protocol["low_torso_occupancy"])
    longest = current = 0
    for value in bad:
        current = current + 1 if value else 0
        longest = max(longest, current)
    reasons = []
    if np.median(area) < protocol["median_mask_bbox_area"]:
        reasons.append("low_median_foreground_support")
    if np.median(core) < protocol["median_torso_occupancy"]:
        reasons.append("low_median_torso_support")
    if bad.mean() > protocol["maximum_low_support_fraction"]:
        reasons.append("frequent_low_foreground_support")
    if longest / 25 > protocol["maximum_low_support_run_seconds"]:
        reasons.append("sustained_low_foreground_support")
    usable = bool(binary.any())
    return {"passed": usable, "preparation_usable": usable,
            "reasons": [] if usable else ["empty_mask_all_frames"],
            "risk_flags": reasons, "occlusion_exclusion": False,
            "final_eligibility": "pending_gvhmr_temporal_and_identity_review",
            "low_support_fraction": float(bad.mean()),
            "longest_low_support_frames": longest, "mask_area_median": float(np.median(area)),
            "torso_median": float(np.median(core)), "mask_area_per_frame": area.tolist(),
            "torso_per_frame": core.tolist(),
            "scope": "Partial occlusion/missing limbs are not exclusions. Occupancy is a risk proxy only.",
            "identity_review": "pending", "visual_review": "pending"}


def screen(root, masks, mapping):
    protocol_path = Path(__file__).resolve().parents[1] / "INPUT_GATE_PROTOCOL.json"
    assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == (
        protocol_path.with_suffix(".sha256")).read_text().strip()
    protocol = json.loads(protocol_path.read_text())
    boxes = np.asarray([x["bbox_input_xyxy"] for x in mapping["frames"]])
    result = evaluate(masks, boxes, protocol["mask"])
    (root / "research/batch_gate_mask.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result
