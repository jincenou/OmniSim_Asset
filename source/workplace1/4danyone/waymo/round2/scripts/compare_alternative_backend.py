"""Match completed alternative-backend videos to existing FTGS cameras/times."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image
import torch
from evaluate_asset import Renderer, sha, save
from render61_helpers import pane, uint8
from video_io import MP4Writer

R = Path(__file__).resolve().parents[1]
W = R.parent
A = R / "alternatives/4dgaussians"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=["batch_0084_twentyfour", "walk063_rootlocal24", "m02_short61_24", "r07_short61_24", "phone_full", "batch_0021_rootlocal24"], required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    physical = int(os.environ["CUDA_VISIBLE_DEVICES"])
    assert physical in (4, 5, 6, 7)
    expected = subprocess.check_output(["nvidia-smi", f"--id={physical}", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip().removeprefix("GPU-")
    torch.cuda.init()
    assert torch.cuda.device_count() == 1 and str(torch.cuda.get_device_properties(0).uuid) == expected
    root = A / "runs" / args.run
    results = json.loads((root / "evaluation/results.json").read_text())
    completion = json.loads((root / "complete.json").read_text())
    assert results["state"] == "new_process_replayed_evaluated_decoded_visual_review_pending"
    manifest = json.loads((A / "data" / args.label / "fdanyone.json").read_text())
    data = Path(manifest["source_data"])
    frames, fps = manifest["frame_count"], manifest["fps"]
    assert frames in (61, 121) and fps == 25
    if frames == 61:
        from evaluate_asset61 import Renderer as ActualRenderer
        base = R / "reconstruction/outputs" / args.label
        if args.label == "m02_short61_24":
            ordinary, optimized = R / "diagnostics/m02_duration/sigma15", R / "diagnostics/m02_duration/sigma8"
        else:
            assert args.label == "r07_short61_24"
            ordinary, optimized = R / "diagnostics/weak_velocity/zero", R / "diagnostics/weak_velocity/zero_sigma8"
        roots = [base, ordinary, optimized]
        names = ["FTGS original5k", "FTGS15k sigma15", "FTGS15k sigma8"]
        paths = [base / "ckpts/ckpt_4999.pt", ordinary / "ckpts/ckpt_14999.pt", optimized / "ckpts/ckpt_14999.pt"]
    elif args.label == "batch_0021_rootlocal24":
        ActualRenderer = Renderer
        base = R / "reconstruction/outputs" / args.label
        roots, names, paths = [base], ["FTGS original5k"], [base / "ckpts/ckpt_4999.pt"]
    else:
        ActualRenderer = Renderer
        base = R / "reconstruction/outputs" / args.label if args.label == "walk063_rootlocal24" else W / "reconstruction/outputs" / args.label
        capacity = R / "quality" / f"{args.label}_capacity50k"
        roots = [base, capacity]
        names = ["FTGS original5k", "FTGS density50k"]
        paths = [base / "ckpts/ckpt_4999.pt", capacity / "ckpts/ckpt_49999.pt"]
    renderers = [ActualRenderer(path, data) for path in paths]
    assert all(np.allclose(renderer.parser.transform, manifest["source_to_normalized"]) for renderer in renderers)
    metric_paths = [path / "quality_metrics/metrics.json" for path in roots]
    if args.label == "phone_full":
        # This baseline was evaluated without moving its original model package.
        metric_paths[0] = R / "quality/baseline_phone_full/quality_metrics/metrics.json"
    metrics = [json.loads(path.read_text()) for path in metric_paths]
    assert all(metric["checkpoint_sha256"] == sha(checkpoint) for metric, checkpoint in zip(metrics, paths, strict=True))
    old = metrics[-1]
    old_hashes = {(row["view"], row["frame"]): row["sha256"] for row in old["inputs"]}
    assert all(old_hashes[row["view"], row["frame"]] == row["input_sha256"] for row in results["samples"])
    assert all(old["crops"][str(view)] == results["crops"][str(view)] for view in (0, 8, 16))
    for metric in metrics:
        assert all(metric["crops"][str(view)] == results["crops"][str(view)] for view in (0, 8, 16))
        hashes = {(row["view"], row["frame"]): row["sha256"] for row in metric["inputs"]}
        assert all(hashes[row["view"], row["frame"]] == row["input_sha256"] for row in results["samples"])
    out = root / "ftgs_comparison"
    out.mkdir(exist_ok=False)
    videos = {}
    with ExitStack() as stack:
        captures = {}
        writers = {}
        for view in (0, 8, 16):
            video = root / "evaluation" / f"view{view:02d}.mp4"
            assert sha(video) == results["videos"][video.name]["sha256"]
            cap = cv2.VideoCapture(str(video))
            assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) == 25
            captures[view] = cap
            stack.callback(cap.release)
            writers[view] = stack.enter_context(MP4Writer(str(out / video.name), fps, 352 * (len(paths) + 2), 664))
        for frame in range(frames):
            for view in (0, 8, 16):
                ok, decoded = captures[view].read()
                assert ok and decoded.shape == (640, 704, 3)
                new_rgb = cv2.cvtColor(decoded[:, 352:], cv2.COLOR_BGR2RGB)
                path = data / "images" / f"cam{view:02d}" / f"{frame:06d}.png"
                rgba = np.asarray(Image.open(path), dtype=np.float32) / 255
                target = rgba[..., :3] * rgba[..., 3:] + 1 - rgba[..., 3:]
                panels = [pane(Image.fromarray(uint8(target)), "Same generated target")]
                for name, renderer in zip(names, renderers, strict=True):
                    raw, alpha = renderer.render(view, frame)
                    assert np.isfinite(raw).all() and np.isfinite(alpha).all()
                    panels.append(pane(Image.fromarray(uint8(np.clip(raw + 1 - alpha[..., None], 0, 1))), name))
                panels.append(pane(Image.fromarray(new_rgb), f"4DGS {completion['coarse_iterations']}+{completion['fine_iterations']}(video)"))
                array = np.concatenate(panels, axis=1)
                writers[view].append_data(array)
                if frame in np.rint(np.linspace(0, frames - 1, 5)).astype(int):
                    Image.fromarray(array).save(out / f"view{view:02d}_frame{frame:03d}.jpg", quality=95)
        assert all(not cap.read()[0] for cap in captures.values())
    for view in (0, 8, 16):
        path = out / f"view{view:02d}.mp4"
        cap = cv2.VideoCapture(str(path))
        count = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            assert image.shape == (664, 352 * (len(paths) + 2), 3)
            count += 1
        cap.release()
        assert count == frames
        videos[path.name] = {"frames": count, "sha256": sha(path)}
    before = old["aggregates"]["validation_generated"]["0.5"]
    after = results["mean"]["0.5"]
    save(out / "validation.json", {"state": "matched_cameras_inputs_crops_rendered_decoded_visual_review_pending",
        "physical_gpu": physical, "cuda_uuid": expected, "videos": videos,
        "gray_ftgs_reference": before, "gray_new_recipe": after,
        **({"gray_ftgs_density50k": before} if "FTGS density50k" in names else {}),
        "gray_ftgs_variants": {name: value["aggregates"]["validation_generated"]["0.5"] for name, value in zip(names, metrics, strict=True)},
        "comparison_baseline": names[-1], "frames": frames,
        "new_minus_ftgs": {key: after[key] - before[key] for key in after},
        "ftgs_checkpoints": {str(path): sha(path) for path in paths},
        "new_results_sha256": sha(root / "evaluation/results.json"),
        "scope": "Complete recipe comparison with different capacity/loss/budget; not representation-only ablation.",
        "display": "New model column decoded from its MP4; quantitative metrics come from direct model rendering, not compressed video.",
        "lpips_implementation": "FTGS TorchMetrics AlexNet vs new lpips0.1.4 AlexNet; same crop/input range; one shared actual-RGBA pair differs by1.19e-7, not exhaustive proof.",
        "lpips_equivalence": str(A / "reports/lpips_equivalence.json")})
    print(json.dumps({"label": args.label, "state": "rendered_decoded_visual_review_pending"}), flush=True)


if __name__ == "__main__":
    main()
