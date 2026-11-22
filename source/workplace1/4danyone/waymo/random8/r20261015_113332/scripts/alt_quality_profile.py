"""Dense train/validation motion profiles and matched-protocol SSIM supplement."""
from argparse import ArgumentParser, Namespace
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageDraw
import torch

from gpu_guard import validate_gpu

A = Path(__file__).resolve().parents[1] / 'four_dg'
sys.path.insert(0, str(A / 'repo'))
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.waymo_generated import GeneratedCameras, read_waymo_generated
from utils.loss_utils import ssim


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(error):
    return -10 * math.log10(max(float(error), 1e-12))


def summary(rows):
    keys = ('full_psnr', 'foreground_psnr', 'lower_region_psnr',
            'lower_target_recall_alpha05', 'whole_target_recall_alpha05',
            'silhouette_iou', 'outside_alpha_mean')
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


@torch.no_grad()
def main():
    parser = ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    device = validate_gpu()
    config = json.loads((args.run / 'experiment.json').read_text())
    complete = json.loads((args.run / 'complete.json').read_text())
    original = json.loads((args.run / 'evaluation/results.json').read_text())
    assert complete['fine_iterations'] == 30000
    _, manifest = read_waymo_generated(config['dataset']['source_path'])
    assert sha(Path(config['dataset']['source_path']) / 'fdanyone.json') == config['input_manifest_sha256']
    heldout, training = [0, 8, 16], [1, 9, 17]
    assert set(training) <= set(manifest['train_views'])
    views = heldout + training
    frames = manifest['frame_count']
    sampled = sorted({row['frame'] for row in original['samples'] if row['background'] == .5})
    assert sampled == np.rint(np.linspace(0, frames - 1, 13)).astype(int).tolist()
    model = GaussianModel(config['dataset']['sh_degree'], Namespace(**config['hidden']))
    saved = args.run / 'point_cloud/iteration_30000'
    model.load_ply(str(saved / 'point_cloud.ply'))
    model.load_model(str(saved))
    model._deformation.eval()
    model_hashes = {name: sha(saved / name) for name in original['asset_files']}
    assert model_hashes == original['asset_files']
    pipeline = Namespace(**config['pipeline'])
    cameras = GeneratedCameras(manifest, views)
    out = A / 'diagnostics/quality_profile_v1' / args.run.name
    out.mkdir(parents=True, exist_ok=False)
    rows, extra = [], []
    black = torch.zeros(3, device='cuda')
    white_points = torch.ones_like(model.get_xyz)
    for frame in range(frames):
        for vi, view in enumerate(views):
            camera = cameras[vi * frames + frame]
            raw = render(camera, model, pipeline, black, stage='fine', cam_type='WaymoGenerated')['render']
            alpha_rgb = render(camera, model, pipeline, black, stage='fine', cam_type='WaymoGenerated',
                               override_color=white_points)['render']
            assert torch.isfinite(raw).all() and torch.isfinite(alpha_rgb).all()
            assert (alpha_rgb.max(0).values - alpha_rgb.min(0).values).max() < 1e-5
            alpha = alpha_rgb[0].clamp(0, 1)
            target_alpha = camera.mask[0].cuda()
            mask = target_alpha > .5
            ys, xs = torch.where(mask)
            assert len(ys) > 100 and (~mask).any()
            lower = mask & (torch.arange(manifest['height'], device='cuda')[:, None] >=
                            ys.min() + .65 * (ys.max() - ys.min()))
            assert lower.any()
            prediction = (raw + (1 - alpha) * .5).clamp(0, 1)
            # The loader stores white-composited RGB; change background once.
            target = camera.original_image.cuda() - (1 - target_alpha)[None] * .5
            error = (prediction - target).square()
            silhouette = alpha > .5
            row = {'view': view, 'frame': frame, 'seconds': frame / manifest['fps'],
                   'split': 'heldout_generated' if view in heldout else 'training_generated',
                   'full_psnr': score(error.mean()), 'foreground_psnr': score(error[:, mask].mean()),
                   'lower_region_psnr': score(error[:, lower].mean()),
                   'lower_target_recall_alpha05': float((alpha[lower] > .5).float().mean()),
                   'whole_target_recall_alpha05': float((alpha[mask] > .5).float().mean()),
                   'silhouette_iou': float((silhouette & mask).sum() / (silhouette | mask).sum()),
                   'outside_alpha_mean': float(alpha[~mask].mean())}
            rows.append(row)
            if view in heldout and frame in sampled:
                x0, y0, x1, y1 = original['crops'][str(view)]
                full = float(ssim(prediction[None], target[None]))
                crop = float(ssim(prediction[None, :, y0:y1, x0:x1], target[None, :, y0:y1, x0:x1]))
                assert math.isfinite(full) and math.isfinite(crop)
                source = Path(manifest['source_data']) / 'images' / f'cam{view:02d}/{frame:06d}.png'
                extra.append({'view': view, 'frame': frame, 'background': .5,
                              'full_ssim': full, 'crop_ssim': crop,
                              'full_psnr': row['full_psnr'], 'foreground_psnr': row['foreground_psnr'],
                              'input_sha256': sha(source)})
        if frame % 20 == 0:
            print(json.dumps({'run': args.run.name, 'frame': frame, 'total_frames': frames}), flush=True)
    assert len(extra) == 39 and len(rows) == frames * 6
    reference_rows = {(row['view'], row['frame']): row for row in original['samples'] if row['background'] == .5}
    differences = []
    for row in extra:
        ref = reference_rows[row['view'], row['frame']]
        assert row['input_sha256'] == ref['input_sha256']
        differences.extend(abs(row[key] - ref[key]) for key in ('full_psnr', 'foreground_psnr'))
    maximum = max(differences)
    assert maximum < .0005, maximum
    worst = {view: min((row for row in rows if row['view'] == view),
                      key=lambda row: row['lower_target_recall_alpha05']) for view in views}
    snapshots = []
    for vi, view in enumerate(views):
        frame = worst[view]['frame']
        camera = cameras[vi * frames + frame]
        raw = render(camera, model, pipeline, black, stage='fine', cam_type='WaymoGenerated')['render']
        alpha = render(camera, model, pipeline, black, stage='fine', cam_type='WaymoGenerated',
                       override_color=white_points)['render'][0].clamp(0, 1)
        target_alpha = camera.mask[0].cuda()
        target = camera.original_image.cuda() - (1 - target_alpha)[None] * .5
        prediction = (raw + (1 - alpha) * .5).clamp(0, 1)
        deficit = (target_alpha - alpha).clamp(0, 1)
        excess = (alpha - target_alpha).clamp(0, 1)
        heat = torch.stack([deficit, torch.zeros_like(deficit), excess])
        pixels = torch.cat([target, prediction, heat], dim=2).permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray(np.round(pixels * 255).astype(np.uint8))
        header = Image.new('RGB', (image.width, image.height + 28), 'white')
        header.paste(image, (0, 28))
        ImageDraw.Draw(header).text((5, 5),
            f'view{view:02d},frame{frame}: generated | reconstruction | alpha deficit red/excess blue', fill='black')
        path = out / f'worst_view{view:02d}_frame{frame:03d}.png'
        header.save(path)
        snapshots.append({'view': view, 'frame': frame, 'path': path.name, 'sha256': sha(path)})
    result = {'updated_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              'state': 'dense_train_heldout_profiles_and_matched_ssim_complete_visual_review_pending',
              'run': str(args.run), 'label': manifest['label'], 'device': device,
              'frame_count': frames, 'heldout_views': heldout, 'diagnostic_train_views': training,
              'samples': rows, 'mean_dense': {split: summary([row for row in rows if row['split'] == split])
                                            for split in ('heldout_generated', 'training_generated')},
              'worst_lower_region_frames': list(worst.values()), 'snapshots': snapshots,
              'ssim_samples': extra,
              'ssim_mean': {key: float(np.mean([row[key] for row in extra])) for key in ('full_ssim', 'crop_ssim')},
              'ssim_protocol': 'Official pinned4DGaussians utils.loss_utils.ssim;window11,sigma1.5,padding5,RGB range[0,1];gray0.5;background;3heldout x13same times;fixed same crops.',
              'matched_existing_psnr_max_delta_db': maximum,
              'original_evaluation_sha256': sha(args.run / 'evaluation/results.json'),
              'asset_sha256': model_hashes,
              'scope': 'Generated observations,not real Waymo novel-view GT. Bottom35% of target bbox is image proxy,not semantic foot GT. Train views1/9/17 are a diagnostic subset,not all21. No point filtering or retraining.',
              'production_approved': False}
    tmp = out / 'results.writing'
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    tmp.replace(out / 'results.json')
    print(json.dumps({'state': result['state'], 'run': args.run.name,
                      'ssim': result['ssim_mean'], 'dense_mean': result['mean_dense']}), flush=True)


if __name__ == '__main__':
    main()
