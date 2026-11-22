"""Terminal projected support with exact occlusion-weighted color partitioning.

No parameter update. A Gaussian's group is a view-dependent reference-core
assignment of its projected center, NOT a semantic label or visible point count.
"""
import argparse
import ast
import csv
import json
import os
import sys
from types import SimpleNamespace
import numpy as np
import torch
from render_baseline import ROOT, BASE, sha, save, protect_old
from gpu_device import verify
from roi_candidates import PARTS


def old_model(label, method, data):
    sys.path.insert(0, str(BASE / 'scripts'))
    path = BASE / 'scripts/evaluate_branch.py'
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ('FT', 'HUST')]
    scope = dict(R=BASE, F=data['frame_count'], H=[0, 8, 16], sys=sys, os=os,
                 json=json, np=np, torch=torch, SimpleNamespace=SimpleNamespace, sha=sha)
    exec(compile(tree, str(path), 'exec'), scope)
    return scope['FT' if method.startswith('ftgs') else 'HUST'](BASE / 'samples' / label, method, data)


def capture_native(module, call, cam):
    original = module.GaussianRasterizer
    saved = {}
    def factory(*args, **kwargs):
        raster = original(*args, **kwargs)
        def invoke(**inputs):
            result = raster(**inputs)
            saved.update(raster=raster, inputs=inputs, result=result)
            return result
        return invoke
    module.GaussianRasterizer = factory
    try:
        call()
    finally:
        module.GaussianRasterizer = original
    means = saved['inputs']['means3D']
    clip = torch.cat([means, torch.ones_like(means[:, :1])], 1) @ cam.full_proj_transform.cuda()
    uv = ((clip[:, :2] / clip[:, 3:] + 1) * torch.tensor([352., 640.], device='cuda') / 2 - .5).cpu().numpy()
    visible = (saved['result'][1] > 0).cpu().numpy()
    opacity = saved['inputs']['opacities'].reshape(-1).cpu().numpy()
    def color_render(colors):
        inputs = dict(saved['inputs'], shs=None, colors_precomp=colors)
        return saved['raster'](**inputs)[0].permute(1, 2, 0).cpu().numpy()
    return uv, visible, opacity, color_render


def capture_ft(model, view, frame):
    import evaluate_asset
    original = evaluate_asset.rasterization
    saved = {}
    def invoke(**kwargs):
        result = original(**kwargs)
        saved.update(inputs=kwargs, meta=result[2])
        return result
    evaluate_asset.rasterization = invoke
    try:
        model.render(view, frame)
    finally:
        evaluate_asset.rasterization = original
    n = len(saved['inputs']['means'])
    meta = saved['meta']
    uv = np.full((n, 2), np.nan, np.float32)
    visible = np.zeros(n, bool)
    if saved['inputs']['packed']:
        ids = meta['gaussian_ids'].cpu().numpy()
        uv[ids] = meta['means2d'].cpu().numpy()
        radii = meta['radii'].cpu().numpy()
        visible[ids] = np.any(radii > 0, axis=-1) if radii.ndim == 2 else radii > 0
    else:
        uv[:] = meta['means2d'][0].cpu().numpy()
        radii = meta['radii'][0].cpu().numpy()
        visible[:] = np.any(radii > 0, axis=-1) if radii.ndim == 2 else radii > 0
    opacity = saved['inputs']['opacities'].reshape(-1).cpu().numpy()
    def color_render(colors):
        return original(**dict(saved['inputs'], colors=colors, sh_degree=None))[0][0].cpu().numpy()
    return uv, visible, opacity, color_render


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--method', required=True)
    args = parser.parse_args()
    device = verify()
    data = json.loads((BASE / 'four_dg/data' / args.label / 'fdanyone.json').read_text())
    os.environ['FDANYONE_FRAMES'] = str(data['frame_count'])
    sys.addaudithook(protect_old)
    out = ROOT / 'audit/support' / args.label / args.method
    out.mkdir(parents=True, exist_ok=False)
    if args.method == 'grid4d_33k':
        import grid_common as grid
        model, deform, pipe, _ = grid.load_asset(ROOT / 'runs' / f'{args.label}_33k_v1', 33000)
    else:
        model = old_model(args.label, args.method, data)
    frames = [0, (data['frame_count'] - 1) // 2, data['frame_count'] - 1]
    rows, checks = [], []
    for view in [0, 8, 16]:
        with np.load(ROOT / 'audit/roi' / args.label / f'view{view:02d}.npz') as z:
            labels = z['labels']
        rawroot = ROOT / 'tmp/grid_raw' / args.label if args.method == 'grid4d_33k' else ROOT / 'tmp/baseline_raw' / args.label / args.method
        with np.load(rawroot / f'view{view:02d}.npz') as z:
            reference_alphas = z['alpha'][frames]
        for j, frame in enumerate(frames):
            if args.method.startswith('ftgs'):
                uv, visible, opacity, renderer = capture_ft(model, view, frame)
            else:
                import gaussian_renderer as native
                if args.method == 'grid4d_33k':
                    cam = grid.camera(data, view, frame)
                    warp = deform.step(model.get_xyz.detach(), cam.fid.unsqueeze(0).expand(len(model.get_xyz), -1))
                    def call():
                        return native.render(cam, model, pipe, torch.zeros(3, device='cuda'), warp['d_xyz'], warp['d_rotation'], warp['d_scaling'])
                else:
                    cam = model.dataset[[0, 8, 16].index(view) * data['frame_count'] + frame]
                    def call():
                        return native.render(cam, model.model, model.pipe, torch.zeros(3, device='cuda'), stage='fine', cam_type='WaymoGenerated')
                uv, visible, opacity, renderer = capture_native(native, call, cam)
            n = len(uv)
            inside = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < 352) & (uv[:, 1] >= 0) & (uv[:, 1] < 640) & visible
            ids = np.where(inside)[0]
            x, y = np.clip(np.rint(uv[ids, 0]).astype(int), 0, 351), np.clip(np.rint(uv[ids, 1]).astype(int), 0, 639)
            groups = np.zeros(n, np.int64)
            groups[ids] = labels[frame, y, x]
            # All points stay in every pass, including zero-color points. This preserves occlusion.
            maps = []
            for start in [0, 3, 6]:
                colors = np.stack([groups == k if k <= 6 else np.zeros(n, bool) for k in range(start, start + 3)], -1)
                maps.append(renderer(torch.tensor(colors, device='cuda', dtype=torch.float32)))
            contribution = np.concatenate(maps, -1)[..., :7]
            alpha = contribution.sum(-1)
            allone = renderer(torch.ones((n, 3), device='cuda'))[..., 0]
            partition_error = float(np.abs(alpha - allone).max())
            replay_error = float(np.abs(alpha - reference_alphas[j]).max())
            assert partition_error < 1e-5 and replay_error < 1e-5, (partition_error, replay_error)
            checks.append(dict(view=view, frame=frame, partition_max_error=partition_error, cache_alpha_max_error=replay_error,
                               points=n, frustum_nonzero_radius_points=int(visible.sum()), unknown_center_points=int((groups == 0).sum())))
            for part_id, part in enumerate(PARTS, 1):
                mask = labels[frame] == part_id
                area = int(mask.sum())
                sel = (groups == part_id) & visible
                mass = contribution[mask].sum(0) if area else None
                rows.append(dict(label=args.label, method=args.method, view=view, frame=frame, part=part, reference_core_pixels=area,
                    center_assigned_points=int(sel.sum()), points_per1000_reference_pixels=float(sel.sum() * 1000 / area) if area else None,
                    mean_center_opacity=float(opacity[sel].mean()) if sel.any() else None,
                    total_alpha_per_reference_pixel=float(mass.sum() / area) if area else None,
                    same_center_group_alpha_per_reference_pixel=float(mass[part_id] / area) if area else None,
                    unknown_center_group_alpha_per_reference_pixel=float(mass[0] / area) if area else None,
                    same_group_fraction_of_alpha=float(mass[part_id] / max(mass.sum(), 1e-12)) if area else None,
                    **{f'contribution_from_group_{k}': float(mass[k] / area) if area else None for k in range(7)}))
    with (out / 'parts.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    save(out / 'complete.json', dict(state='passed', device=device, label=args.label, method=args.method, observations=len(rows), checks=checks,
        definition='View-dependent projected-center group; exact occlusion-weighted alpha contribution via one-hot colors with all geometry retained',
        unknown='Group0 includes unassigned/occluded/offscreen-center points. Gaussian footprints can contribute to other groups.',
        normalization='Point count and contribution mass divided by the same reference-core area; prediction absence is retained',
        limitations='Not semantic Gaussian ownership, not reconstructed geometry truth, not training history; nonzero radius is only frustum/raster support, not proof of unoccluded visibility',
        old_training_history='Sampling/exposure recorded separately; comparable per-step gradients/densification not_recorded; no historical retraining', tolerance=1e-5))
    print(args.label, args.method, 'support checked', flush=True)


if __name__ == '__main__':
    main()
