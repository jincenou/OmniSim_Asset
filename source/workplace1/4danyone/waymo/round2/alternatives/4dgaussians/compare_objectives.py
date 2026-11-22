"""Audit a single alpha-loss change and compare direct-render metrics and videos."""
from argparse import ArgumentParser
from contextlib import ExitStack
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from zoneinfo import ZoneInfo

A = Path(__file__).resolve().parent
R = A.parents[1]
spec = importlib.util.spec_from_file_location("fd_objective_video_io", R / "backend/src/video_io.py")
video_io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video_io)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pane(image, label):
    canvas = Image.new("RGB", (352, 664), (25, 25, 25))
    assert image.size == (352, 640)
    canvas.paste(image, (0, 24))
    ImageDraw.Draw(canvas).text((5, 5), label, fill="white")
    return np.asarray(canvas)


def main():
    parser = ArgumentParser()
    parser.add_argument("--prefix", choices=["m02_61", "r07_61"], required=True)
    requested = parser.parse_args()
    left = A / "runs" / (requested.prefix + "_recipe30k_v1")
    right = A / "runs" / (requested.prefix + "_alpha_recipe30k_v1")
    configs = [json.loads((root / "experiment.json").read_text()) for root in (left, right)]
    metrics = [json.loads((root / "evaluation/results.json").read_text()) for root in (left, right)]
    completions = [json.loads((root / "complete.json").read_text()) for root in (left, right)]
    dataset_left, dataset_right = dict(configs[0]["dataset"]), dict(configs[1]["dataset"])
    assert Path(dataset_left.pop("model_path")).resolve() == left.resolve()
    assert Path(dataset_right.pop("model_path")).resolve() == right.resolve()
    assert dataset_left == dataset_right, "Input dataset configuration changed"
    for key in ("hidden", "pipeline", "seed", "input_manifest_sha256"):
        assert configs[0][key] == configs[1][key], key
    baseline_opt = dict(configs[0]["optimization"])
    assert "lambda_alpha" not in baseline_opt or baseline_opt["lambda_alpha"] == 0
    baseline_opt["lambda_alpha"] = .1
    assert baseline_opt == configs[1]["optimization"], "Unexpected optimization changes"
    assert configs[1]["objective"] == "alpha_l1"
    assert all(item["coarse_iterations"] == 3000 and item["fine_iterations"] == 30000 for item in completions)
    provenance = json.loads((left / "code_provenance.json").read_text())
    variant = json.loads((A / "variants/alpha_l1/provenance.json").read_text())
    assert provenance["sha256"] == variant["source_sha256"] and sha(Path(provenance["snapshot"])) == provenance["sha256"]
    assert configs[1]["trainer_sha256"] == variant["variant_sha256"]
    assert metrics[0]["crops"] == metrics[1]["crops"] and metrics[0]["sampled_frames"] == metrics[1]["sampled_frames"]
    assert [(r["view"], r["frame"], r["background"], r["input_sha256"]) for r in metrics[0]["samples"]] == [
        (r["view"], r["frame"], r["background"], r["input_sha256"]) for r in metrics[1]["samples"]]
    manifest = json.loads((Path(configs[0]["dataset"]["source_path"]) / "fdanyone.json").read_text())
    assert manifest["frame_count"] == 61
    out = A / "comparisons" / (requested.prefix + "_alpha_l1")
    out.mkdir(parents=True, exist_ok=False)
    videos = {}
    with ExitStack() as stack:
        caps, writers = {}, {}
        for view in (0, 8, 16):
            caps[view] = []
            for root, result in zip((left, right), metrics, strict=True):
                path = root / "evaluation" / f"view{view:02d}.mp4"
                assert sha(path) == result["videos"][path.name]["sha256"]
                cap = cv2.VideoCapture(str(path))
                assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) == 25
                stack.callback(cap.release)
                caps[view].append(cap)
            writers[view] = stack.enter_context(video_io.MP4Writer(str(out / f"view{view:02d}.mp4"), 25, 1056, 664))
        for frame in range(61):
            for view in (0, 8, 16):
                path = Path(manifest["source_data"]) / "images" / f"cam{view:02d}" / f"{frame:06d}.png"
                rgba = np.asarray(Image.open(path), dtype=np.float32) / 255
                target = np.round((rgba[..., :3] * rgba[..., 3:] + 1 - rgba[..., 3:]) * 255).astype(np.uint8)
                panels = [pane(Image.fromarray(target), "Same generated target")]
                for cap, label in zip(caps[view], ("4DGS original RGB33k", "4DGS RGB33k+alphaL1(.1)"), strict=True):
                    ok, image = cap.read()
                    assert ok and image.shape == (640, 704, 3)
                    panels.append(pane(Image.fromarray(cv2.cvtColor(image[:, 352:], cv2.COLOR_BGR2RGB)), label))
                combined = np.concatenate(panels, axis=1)
                writers[view].append_data(combined)
                if frame in (0, 15, 30, 45, 60):
                    Image.fromarray(combined).save(out / f"view{view:02d}_frame{frame:03d}.jpg", quality=95)
        assert all(not cap.read()[0] for values in caps.values() for cap in values)
    for view in (0, 8, 16):
        path = out / f"view{view:02d}.mp4"
        cap = cv2.VideoCapture(str(path))
        count = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            assert image.shape == (664, 1056, 3)
            count += 1
        cap.release()
        assert count == 61
        videos[path.name] = {"frames": count, "sha256": sha(path)}
    before, after = metrics[0]["mean"]["0.5"], metrics[1]["mean"]["0.5"]
    report = {"updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "state": "single_alpha_objective_audit_metrics_video_decode_passed_visual_review_pending", "label": manifest["label"],
        "before_gray": before, "after_gray": after,
        "delta": {key: after[key] - before[key] for key in after},
        "lpips_relative_reduction_percent": (before["crop_lpips"] - after["crop_lpips"]) / before["crop_lpips"] * 100,
        "lower_region": {"before": metrics[0]["lower_region"]["mean"], "after": metrics[1]["lower_region"]["mean"]},
        "gaussians": [item["gaussians"] for item in metrics], "videos": videos,
        "inputs_crops_seed_hidden_pipeline_and_budget": "identical; separate validated output model_path excluded",
        "sole_explicit_optimization_change": "Full-image soft alphaL1,lambda0→0.1; adaptive density changes may result from gradients.",
        "evaluation_hashes": [sha(root / "evaluation/results.json") for root in (left, right)],
        "scope": "Generated validation,not real novel-view GT. Video columns decoded; metrics use direct model rendering. No point filtering.",
        "production_approved": False}
    (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"label": manifest["label"], "delta": report["delta"]}), flush=True)


if __name__ == "__main__":
    main()
