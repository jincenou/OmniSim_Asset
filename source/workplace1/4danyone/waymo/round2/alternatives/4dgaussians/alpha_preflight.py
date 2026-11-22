"""Validate soft alpha supervision on the actual short-sequence CUDA renderer."""
from argparse import ArgumentParser
import json
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
from utils.graphics_utils import BasicPointCloud
from gaussian_renderer import render


def main():
    assert not (A / "reports/alpha_preflight.json").exists()
    device = validate_gpu()
    parser = ArgumentParser()
    hidden, pipeline = ModelHiddenParams(parser), PipelineParams(parser)
    args = parser.parse_args([])
    records = []
    for label in ("m02_short61_24", "r07_short61_24"):
        info, manifest = read_waymo_generated(A / "data" / label)
        cloud = info.point_cloud
        subset = BasicPointCloud(cloud.points[:256], cloud.colors[:256], cloud.normals[:256])
        model = GaussianModel(3, hidden.extract(args))
        model._deformation.deformation_net.set_aabb(np.asarray(manifest["motion_aabb_max"]), np.asarray(manifest["motion_aabb_min"]))
        model.create_from_pcd(subset, manifest["scene_radius"], manifest["time_denominator"])
        camera = info.train_cameras[30]
        stages = []
        for stage in ("coarse", "fine"):
            model._opacity.grad = None
            model._xyz.grad = None
            model._deformation.zero_grad(set_to_none=True)
            alpha = render(camera, model, pipeline.extract(args), torch.zeros(3, device="cuda"),
                stage=stage, cam_type="WaymoGenerated", override_color=torch.ones_like(model.get_xyz))["render"][:1].clamp(0, 1)
            target = camera.mask.cuda()
            assert alpha.shape == target.shape == (1, 640, 352)
            loss = torch.abs(alpha - target).mean()
            assert torch.isfinite(loss) and loss > 0
            loss.backward()
            assert model._opacity.grad is not None and torch.isfinite(model._opacity.grad).all()
            assert model._opacity.grad.abs().max() > 0
            assert model._xyz.grad is not None and torch.isfinite(model._xyz.grad).all()
            assert model._xyz.grad.abs().max() > 0
            outside = target < .01
            assert outside.any()
            stages.append({"stage": stage, "full_image_alpha_l1": float(loss),
                "outside_predicted_alpha_sum": float(alpha[outside].sum()),
                "opacity_gradient_max": float(model._opacity.grad.abs().max()),
                "xyz_gradient_max": float(model._xyz.grad.abs().max())})
        records.append({"label": label, "stages": stages})
    result = {"state": "full_image_soft_alpha_forward_backward_passed", "device": device, "records": records,
        "scope": "Actual cameras and256points; no clipping by target mask. Functional alpha test, not full-model quality proof."}
    (A / "reports/alpha_preflight.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
