"""Independent C1 replay and matched-time metrics; no training in this process."""
import argparse
import json
from types import SimpleNamespace
import numpy as np
import torch
import lpips
from PIL import Image, ImageDraw
from grid_common import ROOT, BASE, GaussianModel, camera, manifest, render, save, sha, verify
from evaluate_grid import frozen_functions


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    args = parser.parse_args()
    device = verify()
    run = ROOT / 'mechanism/static/runs' / args.case
    config = json.loads((run / 'config.json').read_text())
    done = json.loads((run / 'complete.json').read_text())
    assert sha(run / 'point_cloud.ply') == done['ply_sha256']
    label, frame = config['label'], config['frame']
    data, _ = manifest(label)
    cameras = json.loads((run / 'camera_data.json').read_text())
    gs = GaussianModel(3)
    gs.load_ply(str(run / 'point_cloud.ply'))
    gs.active_sh_degree = done['active_sh_degree']
    assert len(gs.get_xyz) == done['points']
    pipe = SimpleNamespace(**config['pipeline'])
    background = torch.full((3,), .5, device='cuda')
    errors = []
    with np.load(run / 'reference_render.npz') as refs:
        for key in refs.files:
            actual = render(camera(cameras, int(key), frame), gs, pipe, background, 0., 0., 0.)['render']
            errors.append(float(np.abs(actual.cpu().numpy() - refs[key]).max()))
    assert errors and max(errors) < 1e-5, errors
    scalar, crop, (ssim, ssim_sha) = frozen_functions()
    perceptual = lpips.LPIPS(net='alex').cuda().eval()
    sampled = np.rint(np.linspace(0, data['frame_count'] - 1, 13)).astype(int).tolist()
    assert frame in sampled
    stage = ROOT / 'tmp/static_evaluation' / args.case
    stage.mkdir(parents=True, exist_ok=False)
    rows, blend_errors = [], []
    for view in data['heldout_views']:
        folder = BASE / 'samples' / label / 'data/images' / f'cam{view:02d}'
        box = crop([np.asarray(Image.open(folder / f'{t:06d}.png'))[..., 3] > 127 for t in sampled])
        x0, y0, x1, y1 = box
        rgba = np.asarray(Image.open(folder / f'{frame:06d}.png'), np.float32) / 255
        target = rgba[..., :3] * rgba[..., 3:] + .5 * (1 - rgba[..., 3:])
        mask = rgba[..., 3] > .5
        cam = camera(cameras, view, frame)
        raw = render(cam, gs, pipe, torch.zeros(3, device='cuda'), 0., 0., 0.)['render']
        alpha = render(cam, gs, pipe, torch.zeros(3, device='cuda'), 0., 0., 0., override_color=torch.ones_like(gs.get_xyz))['render'][:1]
        direct = render(cam, gs, pipe, background, 0., 0., 0.)['render']
        blend_errors.append(float((direct - (raw + .5 * (1 - alpha))).abs().max()))
        assert blend_errors[-1] < 1e-5
        predictions = {}
        for name, root in [('HUST_grid61', ROOT / 'tmp/baseline_raw' / label / '4dgs_grid61'),
                           ('Grid4D_33k', ROOT / 'tmp/grid_raw' / label)]:
            assert (root / 'complete.json').exists()
            with np.load(root / f'view{view:02d}.npz') as z:
                a = z['alpha'][frame]
                predictions[name] = (np.clip(z['rgb'][frame] + .5 * (1 - a[..., None]), 0, 1), a)
        raw = raw.permute(1, 2, 0).cpu().numpy()
        alpha = alpha[0].cpu().numpy()
        predictions['Static_3k'] = (np.clip(raw + .5 * (1 - alpha[..., None]), 0, 1), alpha)
        target_tensor = torch.tensor(target.transpose(2, 0, 1).copy(), device='cuda')[None]
        for name, (rgb, a) in predictions.items():
            assert np.isfinite(rgb).all() and np.isfinite(a).all()
            prediction = torch.tensor(rgb.transpose(2, 0, 1).copy(), device='cuda')[None]
            rows.append(dict(method=name, view=view, frame=frame, **scalar(rgb, target, a, mask),
                foreground_soft_coverage=float(a[mask].mean()), ssim=float(ssim(prediction, target_tensor)),
                crop_lpips=float(perceptual(prediction[:, :, y0:y1, x0:x1] * 2 - 1,
                    target_tensor[:, :, y0:y1, x0:x1] * 2 - 1)), crop_xyxy=box))
        strip = Image.new('RGB', (352 * 4, 672), '#282828')
        images = [('Generated reference', target)] + [(k, v[0]) for k, v in predictions.items()]
        draw = ImageDraw.Draw(strip)
        for i, (name, rgb) in enumerate(images):
            strip.paste(Image.fromarray(np.rint(rgb * 255).astype(np.uint8)), (352 * i, 32))
            draw.text((352 * i + 5, 9), f'{name} | {args.case} v{view:02d}', fill='white')
        strip.save(stage / f'view{view:02d}.png')
    fields = ['full_psnr', 'foreground_psnr', 'ssim', 'crop_lpips', 'silhouette_iou', 'outside_alpha_mean', 'foreground_soft_coverage']
    report = dict(state='complete', case=args.case, label=label, frame=frame, device=device,
        independent_replay_max_error=max(errors), background_compositing_max_error=max(blend_errors),
        replay_tolerance=1e-5, samples=rows,
        mean={method: {key: float(np.mean([r[key] for r in rows if r['method'] == method])) for key in fields}
              for method in predictions},
        protocol=dict(heldout_views=data['heldout_views'], background=.5, crop='Frozen per-view union over original13 times',
            reference='Generated-heldout development evaluation, not real Waymo novel-view ground truth', ssim_sha256=ssim_sha,
            comparison='Same three views and one fixed time per case; not compared to the39-pair whole-clip mean',
            interpretation='Static diagnostic only. Different per-time exposure and nonzero-time initialization; not a dynamic asset or matched-compute causal ablation'),
        training=dict(updates=done['updates'], points=done['points'], elapsed_seconds=done['elapsed_seconds'], peak_allocated_bytes=done['peak_allocated_bytes']))
    save(stage / 'metrics.json', report)
    print(json.dumps(dict(case=args.case, mean=report['mean'], replay_error=max(errors))))


if __name__ == '__main__':
    main()
