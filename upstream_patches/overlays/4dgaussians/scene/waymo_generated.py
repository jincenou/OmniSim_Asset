"""Explicit rectangular RGBA loader with a fixed generated-view validation split."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from scene.cameras import Camera
from scene.dataset_readers import SceneInfo
from utils.graphics_utils import BasicPointCloud, focal2fov


class GeneratedCameras(Dataset):
    def __init__(self, manifest, views):
        self.manifest = manifest
        self.cameras = {item["view"]: item for item in manifest["cameras"]}
        self.samples = [(view, frame) for view in views for frame in range(manifest["frame_count"])]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        view, frame = self.samples[index]
        manifest = self.manifest
        path = Path(manifest["source_data"]) / "images" / f"cam{view:02d}" / f"{frame:06d}.png"
        with Image.open(path) as image:
            assert image.mode == "RGBA" and image.size == (manifest["width"], manifest["height"])
            rgba = np.asarray(image, dtype=np.float32) / 255.0
        rgb = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])
        camera = self.cameras[view]
        K = np.asarray(camera["K"], dtype=np.float64)
        w2c = np.linalg.inv(np.asarray(camera["camtoworld"], dtype=np.float64))
        result = Camera(colmap_id=view, R=w2c[:3, :3].T, T=w2c[:3, 3],
            FoVx=focal2fov(K[0, 0], manifest["width"]), FoVy=focal2fov(K[1, 1], manifest["height"]),
            image=torch.from_numpy(rgb.transpose(2, 0, 1).copy()), gt_alpha_mask=None,
            image_name=f"cam{view:02d}/{frame:06d}", uid=view * manifest["frame_count"] + frame,
            data_device="cpu", time=frame / manifest["time_denominator"],
            mask=torch.from_numpy(rgba[..., 3:4].transpose(2, 0, 1).copy()))
        result.source_view = view
        result.source_frame = frame
        # This rasterizer maps NDC to pixels as ((ndc+1)*size-1)/2.
        # Match the saved gsplat K exactly, including its pixel-center offset.
        result.projection_matrix[2, 0] = (2 * K[0, 2] + 1) / manifest["width"] - 1
        result.projection_matrix[2, 1] = (2 * K[1, 2] + 1) / manifest["height"] - 1
        result.full_proj_transform = result.world_view_transform @ result.projection_matrix
        return result


def read_waymo_generated(path):
    root = Path(path)
    manifest = json.loads((root / "fdanyone.json").read_text())
    assert manifest["version"] == 1 and manifest["time_denominator"] == manifest["frame_count"] - 1
    assert set(manifest["train_views"]).isdisjoint(manifest["heldout_views"])
    assert sorted(manifest["train_views"] + manifest["heldout_views"]) == list(range(24))
    with np.load(root / manifest["canonical_npz"]) as values:
        points, colors = values["points"].copy(), values["colors"].copy()
    assert np.isfinite(points).all() and len(points) == manifest["initial_points"]
    cloud = BasicPointCloud(points=points, colors=colors, normals=np.zeros_like(points))
    train = GeneratedCameras(manifest, manifest["train_views"])
    validation = GeneratedCameras(manifest, manifest["heldout_views"])
    video = GeneratedCameras(manifest, [0])
    info = SceneInfo(point_cloud=cloud, train_cameras=train, test_cameras=validation, video_cameras=video,
        nerf_normalization={"radius": manifest["scene_radius"], "translate": np.zeros(3)},
        ply_path=str(root / manifest["canonical_npz"]), maxtime=manifest["frame_count"] - 1)
    return info, manifest
