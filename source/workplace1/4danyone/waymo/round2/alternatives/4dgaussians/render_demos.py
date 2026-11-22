"""Unfiltered temporal, geometry and source-stage views from a reloaded dynamic asset."""
from argparse import ArgumentParser, Namespace
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.spatial.transform import Rotation, Slerp
import torch
from gpu_guard import validate_gpu

A = Path(__file__).resolve().parent
R = A.parents[1]
sys.path.insert(0, str(A / "repo"))
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from scene.waymo_generated import GeneratedCameras, read_waymo_generated
from gaussian_renderer import render
from evaluate import sha

spec = importlib.util.spec_from_file_location("fd_demo_video_io", R / "backend/src/video_io.py")
video_io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video_io)
MP4Writer = video_io.MP4Writer


def uint8(image):
    return np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)


def pane(image, label):
    canvas = Image.new("RGB", (352, 664), (25, 25, 25))
    im = ImageOps.contain(image, (352, 640))
    canvas.paste(im, ((352 - im.width) // 2, 24 + (640 - im.height) // 2))
    ImageDraw.Draw(canvas).text((5, 5), label, fill="white")
    return np.asarray(canvas)


def camera_at(template, pose, time):
    w2c = np.linalg.inv(pose)
    camera = Camera(colmap_id=template.colmap_id, R=w2c[:3, :3].T, T=w2c[:3, 3],
        FoVx=template.FoVx, FoVy=template.FoVy, image=template.original_image,
        gt_alpha_mask=None, image_name="orbit", uid=template.uid, data_device="cpu", time=time)
    camera.projection_matrix = template.projection_matrix.clone()
    camera.full_proj_transform = camera.world_view_transform @ camera.projection_matrix
    return camera


def open_video(path, stack, frames, fps):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) == fps
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == frames
    stack.callback(cap.release)
    return cap


def read_rgb(cap):
    ok, image = cap.read()
    assert ok, "Unexpected video decode end"
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


@torch.no_grad()
def main():
    parser = ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    requested = parser.parse_args()
    device = validate_gpu()
    root = requested.run
    config = json.loads((root / "experiment.json").read_text())
    completion = json.loads((root / "complete.json").read_text())
    evaluation = json.loads((root / "evaluation/results.json").read_text())
    assert evaluation["state"] == "new_process_replayed_evaluated_decoded_visual_review_pending"
    _, manifest = read_waymo_generated(config["dataset"]["source_path"])
    frames, fps = manifest["frame_count"], manifest["fps"]
    width, height = manifest["width"], manifest["height"]
    assert (width, height) == (352, 640)
    job = json.loads(requested.job.read_text())
    assert job["label"] == manifest["label"] and Path(job["data"]).resolve() == Path(manifest["source_data"]).resolve()
    mapping = json.loads(Path(job["mapping"]).read_text())
    native = json.loads(Path(job["native_mapping"]).read_text())
    assert mapping["track_id"] == native["track_id"] and len(mapping["frames"]) == frames
    assert np.allclose([slot["video_seconds"] for slot in mapping["frames"]], np.arange(frames) / fps, atol=1e-8)
    native_by_frame = {item["frame_index"]: item for item in native["frames"]}
    saved = root / f"point_cloud/iteration_{completion['fine_iterations']}"
    assert all(sha(saved / name) == expected for name, expected in evaluation["asset_files"].items())
    model = GaussianModel(config["dataset"]["sh_degree"], Namespace(**config["hidden"]))
    model.load_ply(str(saved / "point_cloud.ply"))
    model.load_model(str(saved))
    model._deformation.eval()
    pipeline = Namespace(**config["pipeline"])
    dataset = GeneratedCameras(manifest, [0, 6, 12])
    out = root / "demo"
    out.mkdir(exist_ok=False)
    cameras = {item["view"]: item for item in manifest["cameras"]}
    poses = np.asarray([cameras[view]["camtoworld"] for view in range(24)])
    closed = np.concatenate([poses, poses[:1]])
    at = np.linspace(0, 24, 72, endpoint=False)
    orbit = np.repeat(np.eye(4)[None], 72, axis=0)
    orbit[:, :3, :3] = Slerp(np.arange(25), Rotation.from_matrix(closed[:, :3, :3]))(at).as_matrix()
    orbit[:, :3, 3] = np.stack([np.interp(at, np.arange(25), closed[:, axis, 3]) for axis in range(3)], axis=-1)
    zero = torch.zeros(3, device="cuda")

    def render_pair(camera):
        raw = render(camera, model, pipeline, zero, stage="fine", cam_type="WaymoGenerated")["render"]
        white = render(camera, model, pipeline, zero, stage="fine", cam_type="WaymoGenerated",
            override_color=torch.ones_like(model.get_xyz))["render"]
        assert torch.isfinite(raw).all() and torch.isfinite(white).all()
        assert (white.max(0).values - white.min(0).values).max() < 1e-5
        return raw.permute(1, 2, 0).cpu().numpy(), white[0].clamp(0, 1).cpu().numpy()[..., None]

    minimum_fixed, minimum_orbit = 10**9, 10**9
    snapshots = [0, (frames - 1) // 2, frames - 1]
    raw_cache, native_hashes, provenance, regions = {}, {}, [], []
    foreground_changes, previous = [], None
    with ExitStack() as stack:
        writers = {name: stack.enter_context(MP4Writer(str(out / (name + ".mp4")), fps, w, h))
            for name, w, h in [("four_panel", 1408, 664), ("fixed_views", 1056, 640),
                ("backgrounds", 1056, 640), ("orbit_fixed_time", 352, 640)]}
        inputs = open_video(Path(job["input"]), stack, frames, fps)
        generated = open_video(Path(job["source"]) / "videos/00.mp4", stack, frames, fps)
        for frame in range(frames):
            pairs = [render_pair(dataset[index * frames + frame]) for index in range(3)]
            counts = [int((alpha > .1).sum()) for _, alpha in pairs]
            assert min(counts) > 100, ("Empty fixed view", frame, counts)
            minimum_fixed = min(minimum_fixed, *counts)
            images = [np.clip(raw + 1 - alpha, 0, 1) for raw, alpha in pairs]
            writers["fixed_views"].append_data(uint8(np.concatenate(images, axis=1)))
            raw, alpha = pairs[0]
            backgrounds = np.concatenate([np.clip(raw + (1 - alpha) * bg, 0, 1) for bg in (0., .5, 1.)], axis=1)
            writers["backgrounds"].append_data(uint8(backgrounds))
            if previous is not None:
                last_raw, last_alpha = previous
                mask = (alpha[..., 0] > .1) | (last_alpha[..., 0] > .1)
                foreground_changes.append(float(np.abs(raw - last_raw)[mask].mean()))
            previous = raw, alpha
            slot = mapping["frames"][frame]
            observation = native_by_frame[slot["left_source_frame"]]
            # Native61 images have already been undistorted/cropped using saved calibration.
            path = Path(observation["path"] if frames == 61 else observation["source_path"])
            if str(path) not in raw_cache:
                native_hashes[str(path)] = sha(path)
                with Image.open(path) as image:
                    crop = image.convert("RGB") if frames == 61 else image.convert("RGB").crop(observation["crop_xyxy"])
                ImageDraw.Draw(crop).rectangle(observation["bbox_input_xyxy"], outline="lime", width=2)
                raw_cache[str(path)] = crop
            panels = [pane(raw_cache[str(path)], "Waymo native crop + box; held"),
                pane(read_rgb(inputs), "Actual input; RIFE + mask"),
                pane(read_rgb(generated), "4DAnyone generated view0"),
                pane(Image.fromarray(uint8(images[0])), "4DGS same generated camera,t")]
            array = np.concatenate(panels, axis=1)
            writers["four_panel"].append_data(array)
            if frame in snapshots:
                Image.fromarray(array).save(out / f"four_panel_frame{frame:03d}.jpg", quality=95)
                Image.fromarray(uint8(backgrounds)).save(out / f"backgrounds_frame{frame:03d}.jpg", quality=95)
                for index, view in enumerate((0, 6, 12)):
                    source = Path(manifest["source_data"]) / "images" / f"cam{view:02d}" / f"{frame:06d}.png"
                    rgba = np.asarray(Image.open(source), dtype=np.float32) / 255
                    ys, xs = np.where(rgba[..., 3] > .5)
                    assert len(xs) > 100
                    y0, y1 = int(ys.min()), int(ys.max()) + 1
                    x0, x1 = max(0, int(xs.min()) - 10), min(width, int(xs.max()) + 11)
                    target = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:]) * .5
                    predicted = pairs[index][0] + (1 - pairs[index][1]) * .5
                    boxes = {"upper_body_accessory_proxy": (x0, int(y0 + .18 * (y1-y0)), x1, int(y0 + .77 * (y1-y0))),
                        "feet_region_proxy": (x0, int(y0 + .72 * (y1-y0)), x1, min(height, y1 + 10))}
                    sheet = Image.new("RGB", (1024, 560), (35, 35, 35))
                    for row, (name, box) in enumerate(boxes.items()):
                        for col, image in enumerate((target, predicted)):
                            tile = ImageOps.contain(Image.fromarray(uint8(image)).crop(box), (500, 250), Image.Resampling.NEAREST)
                            sheet.paste(tile, (col * 512 + (512 - tile.width)//2, row * 280 + 25))
                            ImageDraw.Draw(sheet).text((col * 512 + 5, row * 280 + 5), name + (": generated" if col == 0 else ": reconstructed"), fill="white")
                    sheet.save(out / f"details_view{view:02d}_frame{frame:03d}.jpg", quality=95)
                    regions.append({"view": view, "frame": frame, "boxes": boxes})
            provenance.append({"frame": frame, "seconds": frame / fps, "source_frame": observation["frame_index"],
                "native_path": str(path), "input_interpolated": slot["interpolated"]})
        assert not inputs.read()[0] and not generated.read()[0]
        template = dataset[0]
        for index, pose in enumerate(orbit):
            raw, alpha = render_pair(camera_at(template, pose, .5))
            count = int((alpha > .1).sum())
            assert count > 100, ("Empty orbit", index)
            minimum_orbit = min(minimum_orbit, count)
            image = uint8(raw + 1 - alpha)
            writers["orbit_fixed_time"].append_data(image)
            if index % 12 == 0:
                Image.fromarray(image).save(out / f"orbit_frame{index:03d}.jpg", quality=95)
    videos = {}
    for name, expected in [("four_panel", frames), ("fixed_views", frames), ("backgrounds", frames), ("orbit_fixed_time", 72)]:
        path = out / (name + ".mp4")
        cap = cv2.VideoCapture(str(path))
        assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) == fps
        count = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            assert image.shape[0] == (664 if name == "four_panel" else height)
            count += 1
        cap.release()
        assert count == expected
        videos[path.name] = {"frames": count, "fps": fps, "sha256": sha(path)}
    np.savez_compressed(out / "cameras.npz", orbit_c2w=orbit, orbit_time=np.full(72, .5),
        fixed_c2w=poses[[0, 6, 12]], K=np.asarray(cameras[0]["K"]), fixed_times=np.arange(frames)/(frames-1))
    record = {"state": "rendered_decoded_full_motion_geometry_review_pending", "device": device,
        "videos": videos, "asset_files": evaluation["asset_files"], "evaluation_sha256": sha(root / "evaluation/results.json"),
        "job": str(requested.job), "job_sha256": sha(requested.job), "native_hashes": native_hashes,
        "timeline": provenance, "minimum_fixed_foreground_pixels": minimum_fixed, "minimum_orbit_foreground_pixels": minimum_orbit,
        "foreground_temporal_rgb_mean_abs_changes": foreground_changes, "details": regions,
        "scope": "All Gaussians retained; fixed-time orbit separates geometry from motion. Only generated and reconstructed panels share camera. Native left frame is held at interpolated times; no source-view metric. Regions are mask-relative proxies, nearest enlargement without enhancement.",
        "production_approved": False}
    (out / "validation.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"run": str(root), "state": record["state"], "videos": videos}), flush=True)


if __name__ == "__main__":
    main()
