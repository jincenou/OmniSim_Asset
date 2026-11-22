"""Independent-process replay and fixed generated-observation evaluation."""
from argparse import ArgumentParser, Namespace
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image
import torch
import lpips
from gpu_guard import validate_gpu

R = Path(__file__).resolve().parents[1]
A = R / "four_dg"
sys.path.insert(0, str(A / "repo"))
from scene.gaussian_model import GaussianModel
from scene.waymo_generated import read_waymo_generated
from gaussian_renderer import render
_video_spec = importlib.util.spec_from_file_location("fd_video_io", R / "backend/src/video_io.py")
_video_module = importlib.util.module_from_spec(_video_spec)
_video_spec.loader.exec_module(_video_module)
MP4Writer = _video_module.MP4Writer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def psnr(error):
    return -10 * math.log10(max(float(error), 1e-12))


@torch.no_grad()
def main():
    parser = ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--iteration", type=int, help="Explicit saved fine-stage snapshot; original final evaluation stays unchanged")
    args = parser.parse_args()
    device = validate_gpu()
    config = json.loads((args.run / "experiment.json").read_text())
    completion = json.loads((args.run / "complete.json").read_text())
    _, manifest = read_waymo_generated(config["dataset"]["source_path"])
    frames, fps = manifest["frame_count"], manifest["fps"]
    width, height = manifest["width"], manifest["height"]
    span = (frames - 1) / fps
    assert manifest["time_denominator"] == frames - 1
    assert sha(Path(config["dataset"]["source_path"]) / "fdanyone.json") == config["input_manifest_sha256"]
    from scene.waymo_generated import GeneratedCameras
    dataset = GeneratedCameras(manifest, [0, 8, 16])
    iteration = completion["fine_iterations"] if args.iteration is None else args.iteration
    assert iteration in (5000, 15000, completion["fine_iterations"])
    final = iteration == completion["fine_iterations"]
    saved = args.run / f"point_cloud/iteration_{iteration}"
    for name in ("point_cloud.ply", "deformation.pth", "deformation_table.pth", "deformation_accum.pth"):
        assert (saved / name).is_file(), name
    model = GaussianModel(config["dataset"]["sh_degree"], Namespace(**config["hidden"]))
    model.load_ply(str(saved / "point_cloud.ply"))
    model.load_model(str(saved))
    model._deformation.eval()
    pipeline = Namespace(**config["pipeline"])
    replay_errors = []
    if final:
        reference = np.load(args.run / "reference_render.npz")
        for frame in (0, (frames - 1) // 2, frames - 1):
            raw = render(dataset[frame], model, pipeline, torch.zeros(3, device="cuda"), stage="fine", cam_type="WaymoGenerated")["render"]
            error = float(np.abs(raw.cpu().numpy() - reference[f"view0_frame{frame}"]).max())
            assert error < 1e-5, error
            replay_errors.append(error)
    out = args.run / "evaluation" if final else args.run / f"budget_evaluation/iteration_{iteration}"
    out.mkdir(parents=True, exist_ok=False)
    perceptual = lpips.LPIPS(net="alex").cuda().eval()
    sampled = np.rint(np.linspace(0, frames - 1, 13)).astype(int).tolist()
    assert len(set(sampled)) == 13
    snapshot_frames = np.rint(np.linspace(0, frames - 1, 5)).astype(int).tolist()
    crops = {}
    for vi, view in enumerate((0, 8, 16)):
        masks = [dataset[vi * frames + frame].mask.numpy()[0] > .5 for frame in sampled]
        ys, xs = np.where(np.logical_or.reduce(masks))
        assert len(xs) > 100
        pad = max(4, round(.05 * max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)))
        crops[view] = [max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(width, int(xs.max()) + pad + 1), min(height, int(ys.max()) + pad + 1)]
    rows = []
    lower_rows = []
    snapshots = []
    with ExitStack() as stack:
        writers = {view: stack.enter_context(MP4Writer(str(out / f"view{view:02d}.mp4"), fps, width * 2, height)) for view in (0, 8, 16)}
        for frame in range(frames):
            for vi, view in enumerate((0, 8, 16)):
                camera = dataset[vi * frames + frame]
                raw = render(camera, model, pipeline, torch.zeros(3, device="cuda"), stage="fine", cam_type="WaymoGenerated")["render"]
                white = render(camera, model, pipeline, torch.zeros(3, device="cuda"), stage="fine", cam_type="WaymoGenerated",
                    override_color=torch.ones_like(model.get_xyz))["render"]
                assert torch.isfinite(raw).all() and torch.isfinite(white).all()
                assert (white.max(0).values - white.min(0).values).max() < 1e-5
                alpha = white[0].clamp(0, 1)
                target_mask_all = camera.mask[0].numpy() > .5
                ys, _ = np.where(target_mask_all)
                assert len(ys) > 100
                lower_mask = target_mask_all & (np.arange(height)[:, None] >= ys.min() + .65 * (ys.max() - ys.min()))
                alpha_array = alpha.cpu().numpy()
                lower_rows.append({"view": view, "frame": frame,
                    "lower_target_mean_predicted_alpha": float(alpha_array[lower_mask].mean()),
                    "lower_target_recall_alpha05": float((alpha_array[lower_mask] > .5).mean()),
                    "whole_target_recall_alpha05": float((alpha_array[target_mask_all] > .5).mean())})
                prediction = (raw + 1 - alpha).clamp(0, 1)
                target = camera.original_image.cuda()
                pair = torch.cat([target, prediction], dim=2).permute(1, 2, 0).cpu().numpy()
                writers[view].append_data(np.round(pair * 255).astype(np.uint8))
                if frame in snapshot_frames:
                    path = out / f"view{view:02d}_frame{frame:03d}.jpg"
                    Image.fromarray(np.round(pair * 255).astype(np.uint8)).save(path, quality=95)
                    snapshots.append(path.name)
                if frame in sampled:
                    image_path = Path(manifest["source_data"]) / "images" / f"cam{view:02d}" / f"{frame:06d}.png"
                    rgba = np.asarray(Image.open(image_path), dtype=np.float32) / 255
                    mask = torch.as_tensor(rgba[..., 3] > .5, device="cuda")
                    assert mask.sum() > 100 and (~mask).any()
                    straight = torch.as_tensor(rgba[..., :3].transpose(2, 0, 1).copy(), device="cuda")
                    target_alpha = torch.as_tensor(rgba[..., 3].copy(), device="cuda")
                    x0, y0, x1, y1 = crops[view]
                    for bg in (0., .5, 1.):
                        predicted = (raw + (1 - alpha) * bg).clamp(0, 1)
                        actual_target = straight * target_alpha + (1 - target_alpha) * bg
                        err = (predicted - actual_target).square()
                        score = perceptual(predicted[None, :, y0:y1, x0:x1] * 2 - 1,
                            actual_target[None, :, y0:y1, x0:x1] * 2 - 1).item()
                        silhouette = alpha > .5
                        rows.append({"view": view, "frame": frame, "background": bg, "input_sha256": sha(image_path),
                            "full_psnr": psnr(err.mean()), "foreground_psnr": psnr(err[:, mask].mean()), "crop_lpips": score,
                            "silhouette_iou": float((silhouette & mask).sum() / (silhouette | mask).sum()),
                            "outside_alpha_mean": float(alpha[~mask].mean())})
    videos = {}
    for view in (0, 8, 16):
        path = out / f"view{view:02d}.mp4"
        cap = cv2.VideoCapture(str(path))
        assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) == fps
        count = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            assert image.shape[:2] == (height, width * 2)
            count += 1
        cap.release()
        assert count == frames, count
        videos[path.name] = {"frames": count, "sha256": sha(path)}
    keys = ("full_psnr", "foreground_psnr", "crop_lpips", "silhouette_iou", "outside_alpha_mean")
    result = {"state": "new_process_replayed_evaluated_decoded_visual_review_pending" if final else "snapshot_loaded_evaluated_decoded_training_reference_unavailable", "device": device,
        "gaussians": len(model.get_xyz), "reference": "3 heldout generated views x13 times; not real novel-view GT.",
        "mean": {str(bg): {key: float(np.mean([row[key] for row in rows if row["background"] == bg])) for key in keys} for bg in (0., .5, 1.)},
        "samples": rows, "videos": videos, "snapshots": snapshots, "crops": crops,
        "frame_count": frames, "sampled_frames": sampled, "fps": fps,
        "lower_region": {"definition": "Target-alpha foreground bottom35% of per-frame bbox; image proxy, not semantic foot GT.",
            "frames": frames, "views": [0, 8, 16], "samples": lower_rows,
            "mean": {key: float(np.mean([row[key] for row in lower_rows])) for key in
                ("lower_target_mean_predicted_alpha", "lower_target_recall_alpha05", "whole_target_recall_alpha05")}},
        "new_process_replay_max_abs": max(replay_errors) if final else None,
        "evaluated_fine_iteration": iteration, "total_updates": completion["coarse_iterations"] + iteration,
        "snapshot_scope": "Intermediate snapshot in the same30k learning-rate/density plan,not independent5k/15k recipe. No saved training-time render reference for intermediate snapshots." if not final else "Final independent-process replay validated.",
        "asset_files": {name: sha(saved / name) for name in
            ("point_cloud.ply", "deformation.pth", "deformation_table.pth", "deformation_accum.pth")},
        "protocol": "Same saved RGBA and FTGS cameras; no resizing/filtering; union crop+5%; LPIPS0.1.4 AlexNet; RGB composited once."}
    temporary = out / "results.writing"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(out / "results.json")
    if not final:
        print(json.dumps({"state": result["state"], "iteration": iteration, "gray": result["mean"]["0.5"]}), flush=True)
        return
    relative = saved.relative_to(args.run)
    asset = {"format": "4DGaussians canonical PLY plus neural deformation runtime", "label": manifest["label"],
        "canonical_ply": str(relative / "point_cloud.ply"),
        "deformation_weights": str(relative / "deformation.pth"),
        "deformation_table": str(relative / "deformation_table.pth"),
        "deformation_accum": str(relative / "deformation_accum.pth"),
        "config": "experiment.json", "evaluation": "evaluation/results.json",
        "frames": frames, "fps": fps, "time_units": f"normalized[0,1]; seconds={span}*t",
        "sample_span_seconds": span, "source_to_normalized": manifest["source_to_normalized"],
        "source_data": manifest["source_data"], "source_manifest": str(Path(config["dataset"]["source_path"]) / "fdanyone.json"),
        "source_manifest_sha256": config["input_manifest_sha256"],
        "world_placement": "Waymo metric scale/world/ground alignment not verified",
        "motion_scope": "Replay fitted temporal deformation; arbitrary SMPL-X action retargeting not implemented",
        "gaussians": len(model.get_xyz), "asset_sha256": result["asset_files"],
        "new_process_replay_max_abs": max(replay_errors), "quality": "visual_review_pending_not_production_approved",
        "training": {"coarse_iterations": completion["coarse_iterations"], "fine_iterations": iteration}}
    (args.run / "asset.json").write_text(json.dumps(asset, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"state": result["state"], "gray": result["mean"]["0.5"]}), flush=True)


if __name__ == "__main__":
    main()
