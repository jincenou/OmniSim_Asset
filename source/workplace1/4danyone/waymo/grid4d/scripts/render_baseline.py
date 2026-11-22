"""Render frozen backends into temporary float32 caches for part/time diagnostics."""
import argparse
import ast
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from gpu_device import verify

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'random8/r20261015_113332'
H = [0, 8, 16]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + '.writing')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def protect_old(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = Path(os.fsdecode(args[0])).absolute()
        if path.is_relative_to(BASE):
            flags = args[2]
            assert not (flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)), str(path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--branch', required=True)
    args = parser.parse_args()
    manifest = json.loads((BASE / 'four_dg/data' / args.label / 'fdanyone.json').read_text())
    frames = manifest['frame_count']
    os.environ['FDANYONE_FRAMES'] = str(frames)
    checked_device = verify()
    expected = checked_device['uuid']
    out = ROOT / 'tmp/baseline_raw' / args.label / args.branch
    out.mkdir(parents=True, exist_ok=False)
    sys.addaudithook(protect_old)
    sys.path.insert(0, str(BASE / 'scripts'))
    source = BASE / 'scripts/evaluate_branch.py'
    tree = ast.parse(source.read_text())
    # Reuse exactly the two audited render classes, without executing old cache/output settings.
    tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ('FT', 'HUST')]
    assert len(tree.body) == 2
    scope = dict(R=BASE, F=frames, H=H, sys=sys, os=os, json=json, np=np,
                 torch=torch, SimpleNamespace=SimpleNamespace, sha=sha)
    exec(compile(tree, str(source), 'exec'), scope)
    kind = 'FT' if args.branch.startswith('ftgs') else 'HUST'
    model = scope[kind](BASE / 'samples' / args.label, args.branch, manifest)
    model.validate()
    previous = json.loads((BASE / 'samples' / args.label / 'evaluations' / args.branch / 'results.json').read_text())
    reference = {(r['view'], r['frame']): r for r in previous['samples']}
    errors, files = [], []
    for view in H:
        rgbs, alphas = [], []
        for frame in range(frames):
            rgb, alpha = model.render(view, frame)
            assert rgb.dtype == np.float32 and alpha.dtype == np.float32
            assert np.isfinite(rgb).all() and np.isfinite(alpha).all()
            assert rgb.shape == (640, 352, 3) and alpha.shape == (640, 352)
            rgbs.append(rgb)
            alphas.append(alpha)
            if (view, frame) in reference:
                rgba = np.asarray(Image.open(BASE / 'samples' / args.label / 'data/images' / f'cam{view:02d}' / f'{frame:06d}.png')).astype(np.float32) / 255
                mask = rgba[..., 3] > .5
                target = rgba[..., :3] * rgba[..., 3:] + .5 * (1 - rgba[..., 3:])
                # Match the frozen evaluator's display-range convention exactly.
                # Keep the saved raw RGB/alpha unchanged for later diagnostics.
                prediction = np.clip(rgb + .5 * (1 - alpha[..., None]), 0, 1)
                error = np.square(prediction - target)
                silhouette = alpha > .5
                values = dict(full_psnr=-10 * math.log10(max(float(error.mean()), 1e-12)),
                              foreground_psnr=-10 * math.log10(max(float(error[mask].mean()), 1e-12)),
                              silhouette_iou=float((mask & silhouette).sum() / (mask | silhouette).sum()),
                              outside_alpha_mean=float(alpha[~mask].mean()))
                errors.append(max(abs(values[k] - reference[view, frame][k]) for k in values))
        assert max(errors) < 1e-4, max(errors)
        path = out / f'view{view:02d}.npz'
        np.savez_compressed(path, rgb=np.stack(rgbs), alpha=np.stack(alphas))
        files.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size, frames=frames))
        print(json.dumps(dict(label=args.label, branch=args.branch, view=view, state='raw_cache_written')), flush=True)
    save(out / 'complete.json', dict(state='verified_float32_raw_cache', label=args.label, branch=args.branch,
         device_uuid=expected, pid=os.getpid(), files=files, original_render_source_sha256=sha(source),
         replay=model.replay, original_39_pair_scalar_max_error=max(errors),
         metric_composition='clip(raw_rgb + 0.5 * (1-alpha), 0, 1); raw cache is not clipped',
         old_results_modified=False, temporary=True,
         retention='Delete caches after final part/time diagnostics and complete gallery are verified; source dynamic assets retained.'))


if __name__ == '__main__':
    main()
