"""Read projected means from the actual upstream CUDA rasterizer geometry buffer."""
import itertools
import math
import numpy as np
import torch
from diff_gaussian_rasterization import _C
from grid_common import ROOT, SOURCE, manifest, camera, save, sha, verify, stamp


@torch.no_grad()
def main():
    device = verify()
    rows = []
    for label in ['N04', 'N11']:
        data, _ = manifest(label)
        pixels_depth = np.asarray(list(itertools.product(
            [7.25, data['width'] * .5 + 13.17, data['width'] - 8.6],
            [6.4, data['height'] * .5 - 17.19, data['height'] - 7.7],
            [.45, 1.3, 3.7])), dtype=np.float64)
        for record in data['cameras']:
            cam = camera(data, record['view'], 0)
            intrinsic = np.asarray(record['K'], dtype=np.float64)
            pose = np.asarray(record['camtoworld'], dtype=np.float64)
            depth = pixels_depth[:, 2]
            rays = np.column_stack([pixels_depth[:, :2], np.ones(len(depth))]) @ np.linalg.inv(intrinsic).T
            world = ((rays * depth[:, None]) @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)
            xyz = torch.as_tensor(world, device='cuda')
            n = len(xyz)
            empty = torch.empty(0, device='cuda')
            rotations = torch.zeros(n, 4, device='cuda'); rotations[:, 0] = 1
            result = _C.rasterize_gaussians(
                torch.zeros(3, device='cuda'), xyz, torch.ones(n, 3, device='cuda'),
                torch.full((n, 1), .2, device='cuda'), torch.full((n, 3), .003, device='cuda'),
                rotations, 1., empty, cam.world_view_transform, cam.full_proj_transform,
                math.tan(cam.FoVx / 2), math.tan(cam.FoVy / 2), data['height'], data['width'],
                empty, 0, cam.camera_center, False, False)
            count, rgb, _, radii, geometry, _, _ = result
            assert geometry.dtype == torch.uint8 and geometry.data_ptr() % 128 == 0
            # GeometryState::fromChunk: aligned float depths, bool[3] clamped,
            # int internal_radii, then float2 means2D. ABI is pinned and hashed below.
            offset = 0
            for size in [4 * n, 3 * n, 4 * n]:
                offset = ((offset + 127) // 128) * 128 + size
            offset = ((offset + 127) // 128) * 128
            projected = geometry[offset:offset + n * 8].view(torch.float32).reshape(n, 2).cpu().numpy()
            w2c = np.linalg.inv(pose)
            in_cam = world.astype(np.float64) @ w2c[:3, :3].T + w2c[:3, 3]
            homogeneous = in_cam @ intrinsic.T
            expected = homogeneous[:, :2] / homogeneous[:, 2:]
            error = np.abs(projected - expected)
            assert bool((radii > 0).all()) and count > 0
            assert bool(torch.isfinite(rgb).all()) and float(rgb.sum()) > 0
            maximum = float(error.max())
            rows.append(dict(label=label, view=record['view'], points=n,
                             max_abs_pixel_error=maximum, geometry_buffer_means_offset=offset))
            assert maximum < .001, rows[-1]
    report = dict(state='passed', timestamp=stamp(), device=device,
        scope='Actual native CUDA rasterizer projected means, not just GPU matrix multiplication',
        threshold_pixels=.001, max_abs_pixel_error=max(v['max_abs_pixel_error'] for v in rows),
        cases=rows, image_content_loaded=False,
        abi_sha256=sha(SOURCE / 'submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu'),
        limitations='Projection only; not deformation, training, quality or independent dynamic replay acceptance.')
    save(ROOT / 'research/projection_probe.json', report)
    print({k: v for k, v in report.items() if k != 'cases'}, flush=True)


if __name__ == '__main__':
    main()
