"""Training-only voxel silhouette support at selected failure times, no model updates."""
from argparse import ArgumentParser
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from gpu_guard import validate_gpu

A = Path(__file__).resolve().parent
sys.path.insert(0, str(A / 'repo'))
from scene.waymo_generated import GeneratedCameras, read_waymo_generated


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def projection(points, camera):
    w2c = torch.as_tensor(np.linalg.inv(np.asarray(camera['camtoworld'])), device='cuda', dtype=points.dtype)
    K = torch.as_tensor(camera['K'], device='cuda', dtype=points.dtype)
    cp = points @ w2c[:3, :3].T + w2c[:3, 3]
    image = cp @ K.T
    uv = image[:, :2] / image[:, 2:3]
    return uv, cp[:, 2] > 1e-6


def project_indices(points, camera, width, height):
    uv, front = projection(points, camera)
    pixels = uv.floor().long()
    valid = front & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    idx = (pixels[:, 1].clamp(0, height - 1) * width + pixels[:, 0].clamp(0, width - 1)).to(torch.int32)
    return idx, valid


@torch.no_grad()
def main():
    parser = ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    device = validate_gpu()
    config = json.loads((args.run / 'experiment.json').read_text())
    profile_path = A / 'diagnostics/quality_profile_v1' / args.run.name / 'results.json'
    profile = json.loads(profile_path.read_text())
    _, manifest = read_waymo_generated(config['dataset']['source_path'])
    assert sha(Path(config['dataset']['source_path']) / 'fdanyone.json') == config['input_manifest_sha256']
    train, heldout = manifest['train_views'], manifest['heldout_views']
    assert len(train) == 21 and heldout == [0, 8, 16] and not set(train) & set(heldout)
    width, height = manifest['width'], manifest['height']
    total_frames = manifest['frame_count']
    diagnostic_views = [0, 8, 16, 1, 9, 17]
    frames = sorted({0, (total_frames - 1) // 2, total_frames - 1} |
                    {row['frame'] for row in profile['worst_lower_region_frames']})
    if manifest['label'] == 'walk063_rootlocal24':
        frames = sorted(set(frames) | {80})
    camera_records = {row['view']: row for row in manifest['cameras']}
    loaders = GeneratedCameras(manifest, list(range(24)))
    controls = np.load(Path(config['dataset']['source_path']) / 'canonical.npz')['points'][:256]
    controls = torch.as_tensor(controls, device='cuda', dtype=torch.float32)
    homogeneous = torch.cat([controls, torch.ones(len(controls), 1, device='cuda')], 1)
    projection_errors = []
    for view in range(24):
        camera = loaders[view * total_frames]
        uv, front = projection(controls, camera_records[view])
        clip = homogeneous @ camera.full_proj_transform.cuda()
        ndc = clip[:, :2] / clip[:, 3:4]
        reference = ((ndc + 1) * torch.tensor([width, height], device='cuda') - 1) / 2
        error = float((uv[front] - reference[front]).abs().max())
        assert error < .001, (view, error)
        projection_errors.append(error)
    out = A / 'diagnostics/hull_consistency_v1' / args.run.name
    out.mkdir(parents=True, exist_ok=False)
    low = torch.tensor(manifest['motion_aabb_min'], device='cuda')
    high = torch.tensor(manifest['motion_aabb_max'], device='cuda')
    extent = high - low
    assert (extent > 0).all()
    rows, inputs, saved_masks = [], [], []
    for longest in (160, 224):
        shape = [max(48, round(longest * float(v / extent.max()))) for v in extent]
        axes = [torch.linspace(low[i], high[i], shape[i], device='cuda') for i in range(3)]
        grid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1).reshape(-1, 3)
        projections = {view: project_indices(grid, camera_records[view], width, height) for view in range(24)}
        for frame in frames:
            masks = {}
            for view in range(24):
                source = Path(manifest['source_data']) / 'images' / f'cam{view:02d}/{frame:06d}.png'
                with Image.open(source) as image:
                    assert image.mode == 'RGBA' and image.size == (width, height)
                    mask = np.asarray(image)[..., 3] >= 128
                masks[view] = torch.as_tensor(mask.copy(), device='cuda')
                if longest == 160:
                    inputs.append({'view': view, 'frame': frame, 'sha256': sha(source)})
            votes = torch.zeros(len(grid), dtype=torch.uint8, device='cuda')
            for view in train:
                idx, valid = projections[view]
                votes += (masks[view].flatten()[idx.long()] & valid).to(torch.uint8)
            for threshold in (21, 20):
                occupied = votes >= threshold
                volume = occupied.reshape(shape)
                touches = any(bool(torch.select(volume, axis, edge).any())
                              for axis in range(3) for edge in (0, shape[axis] - 1))
                count = int(occupied.sum())
                for view in diagnostic_views:
                    idx, valid = projections[view]
                    raster = torch.zeros(width * height, device='cuda')
                    raster[idx[occupied & valid].long()] = 1.
                    raster = raster.reshape(height, width)
                    mask = masks[view]
                    ys, _ = torch.where(mask)
                    assert len(ys) > 100
                    lower = mask & (torch.arange(height, device='cuda')[:, None] >=
                                    ys.min() + .65 * (ys.max() - ys.min()))
                    for radius in (0, 2):
                        support = F.max_pool2d(raster[None, None], 2 * radius + 1,
                                              stride=1, padding=radius)[0, 0] > .5
                        rows.append({'frame': frame, 'view': view, 'longest_grid_axis': longest,
                            'grid_shape': shape, 'training_mask_votes_required': threshold,
                            'occupied_voxels': count, 'aabb_boundary_touched': touches,
                            'image_dilation_radius': radius,
                            'whole_target_supported_fraction': float(support[mask].float().mean()),
                            'lower_target_supported_fraction': float(support[lower].float().mean())})
                        selected_worst = next(row['frame'] for row in profile['worst_lower_region_frames'] if row['view'] == view)
                        if longest == 224 and threshold == 20 and radius == 2 and view == 0 and frame == selected_worst:
                            camera = loaders[view * total_frames + frame]
                            target = camera.original_image.cuda() - (1 - camera.mask.cuda()) * .5
                            deficit = (mask & ~support).float()
                            excess = (support & ~mask).float()
                            heat = torch.stack([deficit, torch.zeros_like(deficit), excess])
                            pixels = torch.cat([target, mask[None].float().expand(3, -1, -1),
                                support[None].float().expand(3, -1, -1), heat], dim=2).permute(1, 2, 0).cpu().numpy()
                            image = Image.fromarray(np.round(pixels * 255).astype(np.uint8))
                            header = Image.new('RGB', (image.width, image.height + 28), 'white')
                            header.paste(image, (0, 28))
                            ImageDraw.Draw(header).text((5, 5),
                                f'view00 frame{frame}: generated | alpha | 20/21 hull ray support | deficit red/excess blue', fill='black')
                            path = out / f'view00_frame{frame:03d}_hull_support.png'
                            header.save(path)
                            saved_masks.append({'path': path.name, 'sha256': sha(path)})
            print(json.dumps({'run': args.run.name, 'longest_axis': longest, 'frame': frame}), flush=True)
        del grid, projections
    result = {'updated_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'state': 'training_only_two_resolution_voxel_silhouette_support_complete',
        'run': str(args.run), 'label': manifest['label'], 'device': device, 'frames': frames,
        'train_views': train, 'heldout_views': heldout, 'diagnostic_views': diagnostic_views,
        'rows': rows, 'input_masks': inputs, 'snapshots': saved_masks,
        'projection_max_error_pixels': max(projection_errors),
        'motion_aabb': {'min': manifest['motion_aabb_min'], 'max': manifest['motion_aabb_max']},
        'source_profile_sha256': sha(profile_path), 'source_manifest_sha256': config['input_manifest_sha256'],
        'scope': 'Training21 silhouettes alone define occupied voxels;heldout used only for diagnostic coverage. Static per-time silhouette feasibility proxy,not a reconstructed dynamic asset or real-world GT. Finite voxel resolution and optional2pixel dilation affect coverage. No color/depth/physical motion consistency claim. Selected failure times,not full timeline. AABB is original training-init extent;boundary contact is reported,not silently expanded.',
        'lower_region': 'Target bbox bottom35% foreground image proxy,not semantic foot GT.',
        'model_updated': False, 'production_approved': False}
    (out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'state': result['state'], 'run': args.run.name, 'rows': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
