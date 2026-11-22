"""Index only actual validated alternative-backend states, without changing track counts."""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
A = R / "alternatives/4dgaussians"
RUNS = {"walk063_rootlocal24": ["walk063_recipe30k_v1"],
    "batch_0084_twentyfour": ["0084_recipe30k_v1"],
    "m02_short61_24": ["m02_61_recipe30k_v1", "m02_61_alpha_recipe30k_v1", "m02_61_grid61_recipe30k_v1", "m02_61_smooth001_recipe30k_v1"],
    "r07_short61_24": ["r07_61_recipe30k_v1", "r07_61_alpha_recipe30k_v1", "r07_61_grid61_recipe30k_v1", "r07_61_smooth001_recipe30k_v1"]}
RUNS.update(phone_full=["phone_recipe30k_v1", "phone_grid61_recipe30k_v1"],
            batch_0021_rootlocal24=["walkturn0021_recipe30k_v1", "walkturn0021_grid61_recipe30k_v1"])


def completed_variants(label):
    records = []
    if label not in RUNS:
        return records
    for name in RUNS[label]:
        root = A / "runs" / name
        if not (root / "asset.json").is_file():
            continue
        asset = json.loads((root / "asset.json").read_text())
        metrics = json.loads((root / "evaluation/results.json").read_text())
        assert asset["label"] == label and metrics["state"] == "new_process_replayed_evaluated_decoded_visual_review_pending"
        assert metrics["new_process_replay_max_abs"] < 1e-5
        for key in ("canonical_ply", "deformation_weights", "deformation_table", "deformation_accum"):
            assert (root / asset[key]).is_file()
        demo = root / "demo"
        validated_demo = (demo / "validation.json").is_file()
        if validated_demo:
            proof = json.loads((demo / "validation.json").read_text())
            assert proof["state"] == "rendered_decoded_full_motion_geometry_review_pending"
            assert proof["asset_files"] == metrics["asset_files"]
        title = "4DGS33k RGB"
        if "_alpha_" in name: title = "4DGS33k+alphaL1"
        elif "grid61" in name: title = "4DGS33k temporal grid61"
        elif "smooth001" in name: title = "4DGS33k temporal smoothing0.001"
        records.append({"run": name, "root": root, "asset": asset, "metrics": metrics,
            "title": title,
            "validated_demo": validated_demo})
    return records
