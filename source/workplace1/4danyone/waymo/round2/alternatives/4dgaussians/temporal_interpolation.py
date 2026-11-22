"""Direct temporal residuals on observed slots and50fps model-only interpolation."""
from argparse import ArgumentParser, Namespace
from contextlib import ExitStack
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch
from gpu_guard import validate_gpu

A = Path(__file__).resolve().parent
sys.path.insert(0, str(A / 'repo'))
from scene.gaussian_model import GaussianModel
from scene.waymo_generated import GeneratedCameras, read_waymo_generated
from gaussian_renderer import render
from evaluate import MP4Writer, sha


@torch.no_grad()
def main():
    parser = ArgumentParser()
    parser.add_argument('--label', choices=['m02_short61_24', 'r07_short61_24'], required=True)
    args = parser.parse_args()
    device = validate_gpu()
    prefix = {'m02_short61_24': 'm02_61', 'r07_short61_24': 'r07_61'}[args.label]
    runs = {'default': A / 'runs' / f'{prefix}_recipe30k_v1',
            'grid61': A / 'runs' / f'{prefix}_grid61_recipe30k_v1'}
    out = A / 'diagnostics/temporal_interpolation_v1' / prefix
    out.mkdir(parents=True, exist_ok=False)
    models, pipes, hashes = {}, {}, {}
    for name, root in runs.items():
        config = json.loads((root / 'experiment.json').read_text())
        evaluation = json.loads((root / 'evaluation/results.json').read_text())
        saved = root / 'point_cloud/iteration_30000'
        hashes[name] = {key: sha(saved / key) for key in evaluation['asset_files']}
        assert hashes[name] == evaluation['asset_files']
        model = GaussianModel(config['dataset']['sh_degree'], Namespace(**config['hidden']))
        model.load_ply(str(saved / 'point_cloud.ply'))
        model.load_model(str(saved))
        model._deformation.eval()
        models[name], pipes[name] = model, Namespace(**config['pipeline'])
    default_config = json.loads((runs['default'] / 'experiment.json').read_text())
    grid_config = json.loads((runs['grid61'] / 'experiment.json').read_text())
    assert default_config['input_manifest_sha256'] == grid_config['input_manifest_sha256']
    _, manifest = read_waymo_generated(default_config['dataset']['source_path'])
    frames, views = manifest['frame_count'], [0, 8, 16]
    assert frames == 61 and manifest['fps'] == 25 and manifest['time_denominator'] == 60
    cameras = GeneratedCameras(manifest, views)
    original = json.loads((runs['default'] / 'evaluation/results.json').read_text())
    previous, increments, areas = {}, [], []
    background = torch.zeros(3, device='cuda')
    overrides = {name: torch.ones_like(model.get_xyz) for name, model in models.items()}
    with ExitStack() as stack:
        writers = {view: stack.enter_context(MP4Writer(str(out / f'view{view:02d}_50fps.mp4'),
                        50, 704, 640)) for view in views}
        for half_step in range(2 * (frames - 1) + 1):
            native_frame = half_step // 2
            for vi, view in enumerate(views):
                camera = cameras[vi * frames + native_frame]
                camera.time = half_step / (2 * (frames - 1))
                target_alpha = camera.mask[0].cuda()
                target = camera.original_image.cuda() - (1 - target_alpha)[None] * .5
                x0, y0, x1, y1 = original['crops'][str(view)]
                predictions = []
                for name, model in models.items():
                    raw = render(camera, model, pipes[name], background, stage='fine', cam_type='WaymoGenerated')['render']
                    alpha_rgb = render(camera, model, pipes[name], background, stage='fine', cam_type='WaymoGenerated',
                                       override_color=overrides[name])['render']
                    assert torch.isfinite(raw).all() and torch.isfinite(alpha_rgb).all()
                    alpha = alpha_rgb[0].clamp(0, 1)
                    pred = (raw + (1 - alpha) * .5).clamp(0, 1)
                    area = int((alpha > .5).sum())
                    assert area > 100
                    areas.append({'model': name, 'view': view, 'half_step': half_step, 'foreground_pixels_alpha05': area})
                    predictions.append(pred)
                    if half_step % 2 == 0:
                        key = name, view
                        if key in previous:
                            old_pred, old_alpha, old_target, old_target_alpha = previous[key]
                            rgb_residual = ((pred - old_pred) - (target - old_target))[:, y0:y1, x0:x1].abs().mean()
                            alpha_residual = ((alpha - old_alpha) - (target_alpha - old_target_alpha))[y0:y1, x0:x1].abs().mean()
                            increments.append({'model': name, 'view': view, 'frame': native_frame,
                                'crop_rgb_increment_residual': float(rgb_residual),
                                'crop_alpha_increment_residual': float(alpha_residual)})
                        previous[key] = pred, alpha, target, target_alpha
                pair = torch.cat(predictions, dim=2).permute(1, 2, 0).cpu().numpy()
                writers[view].append_data(np.round(pair * 255).astype(np.uint8))
            if half_step % 30 == 0:
                print(json.dumps({'label': args.label, 'half_step': half_step, 'total': 121}), flush=True)
    video_records = {}
    for view in views:
        path = out / f'view{view:02d}_50fps.mp4'
        cap = cv2.VideoCapture(str(path))
        assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) == 50
        count = 0
        while True:
            ok, pixels = cap.read()
            if not ok:
                break
            assert pixels.shape == (640, 704, 3)
            count += 1
        cap.release()
        assert count == 121
        video_records[path.name] = {'frames': count, 'fps': 50, 'sha256': sha(path)}
    result = {'updated_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'state': 'two_model_direct_temporal_residuals_and_three_50fps_interpolation_videos_complete',
        'label': args.label, 'device': device, 'model_hashes': hashes, 'increments': increments,
        'means': {name: {key: float(np.mean([r[key] for r in increments if r['model'] == name]))
                        for key in ('crop_rgb_increment_residual', 'crop_alpha_increment_residual')}
                  for name in models}, 'alpha_area': areas, 'videos': video_records,
        'source_generated_slots': 61, 'model_render_slots': 121, 'render_fps': 50,
        'sample_span_seconds': 2.4,
        'scope': 'Video columns default/grid61;both model renders on gray0.5. Half-time slots have no observationGT and are explicitly model interpolation,not extra captured/generated frames. Increment residual compares direct25fps model changes with same generated RGB/alpha changes in fixed saved crops;proxy,not physical gait or paper metric. Full model,no pointfilter.',
        'production_approved': False}
    (out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'label': args.label, 'means': result['means']}), flush=True)


if __name__ == '__main__':
    main()
