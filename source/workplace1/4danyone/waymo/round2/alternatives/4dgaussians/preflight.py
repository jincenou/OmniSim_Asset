"""Real rectangular-camera projection and CUDA forward/backward validation."""
from argparse import ArgumentParser
import json
import hashlib
from pathlib import Path
import sys
import numpy as np
import torch
from gpu_guard import validate_gpu

A = Path(__file__).resolve().parent
sys.path.insert(0, str(A / "repo"))
from arguments import ModelHiddenParams, PipelineParams
from scene.gaussian_model import GaussianModel
from scene.waymo_generated import read_waymo_generated
from gaussian_renderer import render
from simple_knn._C import distCUDA2


def main():
    cli = ArgumentParser()
    cli.add_argument("--labels", nargs="+", default=["batch_0084_twentyfour", "walk063_rootlocal24"])
    cli.add_argument("--output", type=Path, default=A / "reports/runtime_preflight.json")
    requested = cli.parse_args()
    assert not requested.output.exists(), "Preserve prior preflight evidence"
    device = validate_gpu()
    parser = ArgumentParser()
    hp, pp = ModelHiddenParams(parser), PipelineParams(parser)
    args = parser.parse_args([])
    observations = []
    for label in requested.labels:
        info, manifest = read_waymo_generated(A / "data" / label)
        frames = manifest["frame_count"]
        assert len(info.train_cameras) == 21 * frames and len(info.test_cameras) == 3 * frames
        assert set(view for view, _ in info.train_cameras.samples).isdisjoint({0, 8, 16})
        assert info.test_cameras[0].time == 0 and info.test_cameras[frames - 1].time == 1
        errors = []
        for index in (0, frames, frames * 2):
            camera = info.test_cameras[index]
            item = next(x for x in manifest["cameras"] if x["view"] == camera.source_view)
            K = np.asarray(item["K"])
            xyz = info.point_cloud.points[::max(1, len(info.point_cloud.points) // 100)]
            homogeneous = np.column_stack([xyz, np.ones(len(xyz))])
            w2c = np.linalg.inv(np.asarray(item["camtoworld"]))
            camera_xyz = homogeneous @ w2c.T
            expected = camera_xyz[:, :3] @ K.T
            expected = expected[:, :2] / expected[:, 2:3]
            clip = homogeneous @ camera.full_proj_transform.numpy()
            ndc = clip[:, :2] / clip[:, 3:4]
            actual = (ndc + 1) * np.array([manifest["width"], manifest["height"]]) / 2 - .5
            error = float(np.abs(actual - expected).max())
            assert error < .001, error
            errors.append(error)
        cloud = info.point_cloud
        from utils.graphics_utils import BasicPointCloud
        subset = BasicPointCloud(cloud.points[:256], cloud.colors[:256], cloud.normals[:256])
        assert torch.isfinite(distCUDA2(torch.as_tensor(subset.points, device="cuda"))).all()
        model = GaussianModel(3, hp.extract(args))
        model._deformation.deformation_net.set_aabb(np.asarray(manifest["motion_aabb_max"]), np.asarray(manifest["motion_aabb_min"]))
        model.create_from_pcd(subset, manifest["scene_radius"], frames - 1)
        camera = info.train_cameras[(frames - 1) // 2]
        stages = []
        for stage in ("coarse", "fine"):
            model._xyz.grad = None
            model._deformation.zero_grad(set_to_none=True)
            package = render(camera, model, pp.extract(args), torch.ones(3, device="cuda"), stage=stage, cam_type="WaymoGenerated")
            image = package["render"]
            assert image.shape == (3, 640, 352) and torch.isfinite(image).all()
            assert package["visibility_filter"].any(), "No visible Gaussian in real camera"
            loss = (image - camera.original_image.cuda()).square().mean()
            loss.backward()
            assert model._xyz.grad is not None and torch.isfinite(model._xyz.grad).all()
            assert model._xyz.grad.abs().max() > 0
            assert package["viewspace_points"].grad is not None
            stages.append({"stage": stage, "loss": float(loss), "xyz_gradient_max": float(model._xyz.grad.abs().max()),
                "visible": int(package["visibility_filter"].sum())})
        observations.append({"label": label, "initial_points": len(cloud.points), "projection_max_error_px": max(errors),
            "frame_count": frames, "sample_span_seconds": (frames - 1) / manifest["fps"],
            "manifest_sha256": hashlib.sha256((A / "data" / label / "fdanyone.json").read_bytes()).hexdigest(),
            "train_observations": len(info.train_cameras), "heldout_observations": len(info.test_cameras),
            "time_endpoints": [0, 1], "cuda_stages": stages})
    report = {"state": "camera_time_knn_rasterizer_forward_backward_passed", "device": device, "observations": observations,
        "scope": "Real cameras and256-point subset; does not prove full optimization quality."}
    requested.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
