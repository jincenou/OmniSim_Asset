"""Complete diagnostic sheets, fixed-rule worst segments and time heatmaps."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from roi_candidates import ROOT, BASE, PARTS
from render_baseline import save, sha

OLD = ['ftgs_base5k', 'ftgs_opt15k', '4dgs_grid25', '4dgs_grid61']
NAMES = dict(ftgs_base5k='FTGS5k', ftgs_opt15k='FTGS15k', **{'4dgs_grid25':'HUST25', '4dgs_grid61':'HUST61'}, grid4d_33k='Grid4D33k')


def fitted(rgb, width, height):
    picture = Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    picture = ImageOps.contain(picture, (width, height), Image.Resampling.BILINEAR)
    tile = Image.new('RGB', (width, height), '#404040')
    tile.paste(picture, ((width - picture.width) // 2, (height - picture.height) // 2))
    return tile


def error_image(pred, target):
    value = np.mean(np.abs(pred - target), -1) / .25
    return plt.get_cmap('magma')(np.clip(value, 0, 1))[..., :3]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--label', required=True); args = parser.parse_args()
    cv2.setNumThreads(2); label = args.label
    methods = OLD + (['grid4d_33k'] if label in ['N04', 'N11'] else [])
    meta = json.loads((BASE / 'four_dg/data' / label / 'fdanyone.json').read_text()); frames = meta['frame_count']
    out = ROOT / 'tmp/gallery' / label; out.mkdir(parents=True, exist_ok=False)
    metric = {}; matrices = {}
    for method in methods:
        rows = list(csv.DictReader((ROOT / 'audit/metrics' / label / f'{method}.csv').open()))
        metric[method] = {(int(r['view']), int(r['frame']), r['part']): r for r in rows}
        matrices[method] = {}
        for field in ['coverage', 'mse', 'temporal_mse']:
            a = np.full((18, frames), np.nan)
            for i, view in enumerate([0, 8, 16]):
                for j, part in enumerate(PARTS):
                    for frame in range(frames):
                        value = metric[method][view, frame, part][field]
                        if value: a[i * 6 + j, frame] = float(value)
            matrices[method][field] = a
    # Freeze explicit selection rule: worst25-frame window by mean core MSE,
    # averaged equally over the four HISTORICAL backends; at least20 valid slots.
    choices = []
    for p, part in enumerate(PARTS):
        best = None
        for vindex, view in enumerate([0, 8, 16]):
            values = np.stack([matrices[m]['mse'][vindex * 6 + p] for m in OLD]).mean(0)
            for start in range(frames - 24):
                window = values[start:start + 25]
                if np.isfinite(window).sum() < 20: continue
                score = float(window[np.isfinite(window)].mean())
                if best is None or score > best['score']:
                    good = np.where(np.isfinite(window))[0]
                    middle = start + int(good[np.argmin(np.abs(good - 12))])
                    best = dict(part=part, view=view, start=start, end=start + 24, representative_frame=middle, score=score)
        choices.append(best or dict(part=part, state='unknown_no_reliable25frame_window'))
    worst = max([c for c in choices if 'score' in c], key=lambda c:c['score'])
    masks = {}; targets = {}; predictions = {}
    for view in sorted({c['view'] for c in choices if 'view' in c}):
        with np.load(ROOT / 'audit/roi' / label / f'view{view:02d}.npz') as z: masks[view] = z['labels']
        needed = {c['representative_frame'] for c in choices if c.get('view') == view}
        if view == worst['view']: needed.update(range(worst['start'], worst['end'] + 1))
        for frame in needed:
            rgba = np.asarray(Image.open(BASE / 'samples' / label / 'data/images' / f'cam{view:02d}' / f'{frame:06d}.png'), np.float32) / 255
            targets[view, frame] = rgba[..., :3] * rgba[..., 3:] + .5 * (1 - rgba[..., 3:])
        for method in methods:
            folder = ROOT / 'tmp/grid_raw' / label if method == 'grid4d_33k' else ROOT / 'tmp/baseline_raw' / label / method
            with np.load(folder / f'view{view:02d}.npz') as z:
                rgb, alpha = z['rgb'], z['alpha']
                for frame in needed:
                    a = alpha[frame].copy()
                    predictions[method, view, frame] = (np.clip(rgb[frame] + .5 * (1 - a[..., None]), 0, 1), a)
    for c in choices:
        if 'view' not in c: continue
        part_id = PARTS.index(c['part']) + 1
        visible = np.any(masks[c['view']][c['start']:c['end'] + 1] == part_id, axis=0)
        y, x = np.where(visible)
        c['crop_xyxy'] = [max(0, int(x.min()) - 12), max(0, int(y.min()) - 12), min(352, int(x.max()) + 13), min(640, int(y.max()) + 13)]
    cols = 1 + 3 * len(methods)
    sheet = Image.new('RGB', (cols * 128, 6 * 234 + 40), '#282828'); draw = ImageDraw.Draw(sheet)
    draw.text((5, 8), f'{label} | Reference-defined crops. RGB / alpha / abs error (0..0.25). Display resized; metrics native.', fill='white')
    for p, c in enumerate(choices):
        ybase = 40 + p * 234
        if 'view' not in c:
            draw.text((5, ybase + 10), c['part'] + ': UNKNOWN; no reliable core window', fill='white'); continue
        view, frame = c['view'], c['representative_frame']; x0, y0, x1, y1 = c['crop_xyxy']
        target = targets[view, frame]
        sheet.paste(fitted(target[y0:y1, x0:x1], 128, 200), (0, ybase + 30))
        draw.text((3, ybase + 2), f'{c["part"]} v{view}', fill='white'); draw.text((3, ybase + 15), f'f{frame} reference', fill='white')
        for m, method in enumerate(methods):
            pred, a = predictions[method, view, frame]
            tiles = [pred, np.repeat(a[..., None], 3, -1), error_image(pred, target)]
            for j, (name, rgb) in enumerate(zip(['RGB', 'alpha', 'error'], tiles)):
                xbase = (1 + 3 * m + j) * 128
                sheet.paste(fitted(rgb[y0:y1, x0:x1], 128, 200), (xbase, ybase + 30))
                draw.text((xbase + 3, ybase + 6), NAMES[method] + ' ' + name, fill='white')
    sheet.save(out / 'parts.jpg', quality=95)
    fig, axes = plt.subplots(len(methods), 3, figsize=(15, 3.1 * len(methods)), squeeze=False, layout='constrained')
    for row, method in enumerate(methods):
        for col, (field, title, vmax) in enumerate([('coverage', 'Soft coverage (higher)', 1.), ('mse', 'RGB RMSE (lower)', .25), ('temporal_mse', 'Motion-compensated RMSE (lower)', .25)]):
            a = matrices[method][field]
            if field != 'coverage': a = np.sqrt(a)
            cmap = plt.get_cmap('viridis' if field == 'coverage' else 'magma').copy(); cmap.set_bad('#777777')
            ax = axes[row, col]; image = ax.imshow(a, aspect='auto', vmin=0, vmax=vmax, cmap=cmap, interpolation='nearest')
            ax.set_title(NAMES[method] + ' | ' + title, fontsize=9)
            ax.set_yticks(range(18), [f'v{v:02d} {part}' for v in [0, 8, 16] for part in PARTS], fontsize=6)
            ax.set_xlabel('frame; gray = unknown/unreliable correspondence', fontsize=7)
            fig.colorbar(image, ax=ax, fraction=.03, pad=.02)
    fig.savefig(out / 'time_metrics.png', dpi=140); plt.close(fig)
    spec = importlib.util.spec_from_file_location('frozen_video', BASE / 'backend/src/video_io.py')
    video = importlib.util.module_from_spec(spec); spec.loader.exec_module(video)
    view = worst['view']; x0, y0, x1, y1 = worst['crop_xyxy']
    width, height = 896, 256 * len(methods)
    writer = video.MP4Writer(str(out / 'worst_segment.mp4'), 25, width, height)
    try:
        for frame in range(worst['start'], worst['end'] + 1):
            canvas = Image.new('RGB', (width, height), '#282828'); ink = ImageDraw.Draw(canvas)
            target = targets[view, frame]
            for row, method in enumerate(methods):
                pred, a = predictions[method, view, frame]
                images = [target, pred, np.repeat(a[..., None], 3, -1), error_image(pred, target)]
                for col, (name, rgb) in enumerate(zip(['Reference', NAMES[method], 'alpha', 'error0..0.25'], images)):
                    canvas.paste(fitted(rgb[y0:y1, x0:x1], 224, 224), (224 * col, 256 * row + 32))
                    ink.text((224 * col + 4, 256 * row + 8), f'{name} {label} {worst["part"]} v{view} f{frame}', fill='white')
            writer.append_data(np.asarray(canvas))
    finally: writer.close()
    cap = cv2.VideoCapture(str(out / 'worst_segment.mp4')); count = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        assert frame.shape == (height, width, 3); count += 1
    cap.release(); assert count == 25
    save(out / 'selection.json', dict(label=label, methods=methods, parts=choices, worst=worst,
        rule='25-frame window maximizing equally weighted mean core MSE of all4 historical methods; minimum20 reliable frames. Per-part/view then overall maximum; ties favor first. Representative closest valid frame to window midpoint.',
        crops='Reference core union over selected25 slots +12px; same box for all methods; predictions not masked or cleaned',
        scales=dict(alpha=[0, 1], rgb_abs_error=[0, .25], heatmap_rgb_rmse=[0, .25], heatmap_temporal_rmse=[0, .25]),
        limitations='One-second worst segment is a diagnostic excerpt, not a full dynamic clip. Estimated cores may omit occluded parts; raw full comparisons remain linked.'))
    save(out / 'complete.json', dict(state='complete', frames=count, files={f.name:sha(f) for f in out.iterdir() if f.is_file()}))
    destination = ROOT / 'gallery' / label; destination.parent.mkdir(exist_ok=True); assert not destination.exists(); out.rename(destination)
    print(label, 'complete diagnostic gallery files', flush=True)


if __name__ == '__main__': main()
