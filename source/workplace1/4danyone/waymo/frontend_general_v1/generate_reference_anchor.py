"""G3: same G2 conditions and four RCP references, constrained during target sampling."""
import importlib
import json
import os
from pathlib import Path
import sys

G = Path(__file__).resolve().parent
sys.path.insert(0, os.environ["FDANYONE_RUNTIME"])


def main():
    from reference_anchor import apply_target_anchors
    import generate_dev6
    module = importlib.import_module("fdanyone.model.inference")
    original_rcp = module._denoise_rcp
    original_targets = module._denoise_targets_single
    captured = {}

    def remember_rcp(**kwargs):
        result = original_rcp(**kwargs)
        captured["latents"] = result
        captured["camera_ids"] = tuple(kwargs["camera_ids"])
        return result

    def targets(**kwargs):
        result, report = apply_target_anchors(
            module, original_targets, kwargs, captured["latents"], captured["camera_ids"])
        (Path(os.environ["FDANYONE_WAYMO_ROOT"]) / "anchor_report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n")
        return result

    module._denoise_rcp = remember_rcp
    module._denoise_targets_single = targets
    generate_dev6.main()
    evidence = Path(os.environ["FDANYONE_WAYMO_ROOT"]) / "stage_evidence/report.json"
    data = json.loads(evidence.read_text())
    data.update(target_denoising_unchanged=False,
                intervention="existing_four_rcp_target_anchors_v1",
                original_four_reference_packing_unchanged=True)
    evidence.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
