"""Full-time matched cameras plus lower-body alpha diagnostics for M02."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
import imageio.v2 as imageio
import torch
from evaluate_asset61 import Renderer, sha, save
from render61_helpers import pane, uint8
from video_io import MP4Writer

R = Path(__file__).resolve().parents[1]

def main():
    root = R/'diagnostics/m02_duration'
    assert json.loads((root/'paired_audit.json').read_text())['state'] == 'only_initial_durations_changed'
    gpu = int(os.environ['CUDA_VISIBLE_DEVICES'])
    assert gpu in (4, 5, 6, 7)
    uuid = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=uuid',
                                    '--format=csv,noheader'], text=True).strip().removeprefix('GPU-')
    torch.cuda.init()
    assert torch.cuda.device_count() == 1 and str(torch.cuda.get_device_properties(0).uuid) == uuid
    paths = [('old5k', R/'reconstruction/outputs/m02_short61_24/ckpts/ckpt_4999.pt'),
             ('sigma15_15k', root/'sigma15/ckpts/ckpt_14999.pt'),
             ('sigma8_15k', root/'sigma8/ckpts/ckpt_14999.pt')]
    data = R/'reconstruction/data/m02_short61_24'
    models = [Renderer(path, data) for _, path in paths]
    assert all(np.array_equal(model.parser.transform, models[0].parser.transform) for model in models)
    out = root/'comparison'
    out.mkdir(exist_ok=False)
    rows = []
    with ExitStack() as stack:
        writers = {view: MP4Writer(str(out/f'view{view:02d}.mp4'), 25, 1408, 664) for view in (0, 8, 16)}
        for writer in writers.values():
            stack.callback(writer.close)
        for frame in range(61):
            for view in writers:
                path = data/'images'/f'cam{view:02d}'/f'{frame:06d}.png'
                with Image.open(path) as image:
                    rgba = np.asarray(image.convert('RGBA'), dtype=np.float32)/255
                mask = rgba[..., 3] > .5
                ys, xs = np.where(mask)
                assert len(ys) > 100
                # A reproducible image-space region, not a semantic foot/leg annotation.
                lower = mask & (np.arange(mask.shape[0])[:, None] >= ys.min()+.65*(ys.max()-ys.min()))
                assert lower.sum() > 10
                target = rgba[..., :3]*rgba[..., 3:] + 1-rgba[..., 3:]
                panels = [pane(Image.fromarray(uint8(target)), 'Generated target + fixed mask')]
                alpha_panels = [pane(Image.fromarray(uint8(np.repeat(rgba[..., 3:], 3, axis=2))), 'Target alpha')]
                for (name, _), model in zip(paths, models, strict=True):
                    rgb, alpha = model.render(view, frame)
                    assert np.isfinite(rgb).all() and np.isfinite(alpha).all()
                    panels.append(pane(Image.fromarray(uint8(np.clip(rgb+1-alpha[..., None], 0, 1))), name))
                    alpha_panels.append(pane(Image.fromarray(uint8(np.repeat(alpha[..., None], 3, axis=2))), name+' alpha'))
                    rows.append({'view': view, 'frame': frame, 'model': name, 'target_sha256': sha(path),
                        'lower_target_pixels': int(lower.sum()),
                        'lower_target_mean_predicted_alpha': float(alpha[lower].mean()),
                        'lower_target_recall_alpha05': float((alpha[lower] > .5).mean()),
                        'whole_target_recall_alpha05': float((alpha[mask] > .5).mean())})
                array = np.concatenate(panels, axis=1)
                writers[view].append_data(array)
                if frame in (0, 10, 20, 30, 40, 50, 60):
                    Image.fromarray(array).save(out/f'view{view:02d}_frame{frame:03d}.jpg', quality=95)
                    Image.fromarray(np.concatenate(alpha_panels, axis=1)).save(out/f'alpha_view{view:02d}_frame{frame:03d}.png')
    videos = {}
    for view in (0, 8, 16):
        path = out/f'view{view:02d}.mp4'
        with imageio.get_reader(str(path), input_params=['-threads', '1']) as reader:
            meta = reader.get_meta_data()
            count = sum(1 for _ in reader)
        assert count == 61 and meta['fps'] == 25 and tuple(meta['size']) == (1408, 664)
        videos[path.name] = {'frames': count, 'fps': 25, 'sha256': sha(path)}
    summary = {name: {key: float(np.mean([row[key] for row in rows if row['model'] == name]))
                for key in ('lower_target_mean_predicted_alpha', 'lower_target_recall_alpha05', 'whole_target_recall_alpha05')}
                for name, _ in paths}
    save(out/'lower_region.json', {'definition': 'Target-alpha foreground in bottom35% of its per-frame bbox height. Image-space lower-body proxy, not semantic foot GT.',
         'reference': 'Generated masked images, not real novel-view GT.', 'rows': rows, 'summary': summary})
    save(out/'validation.json', {'state': 'rendered_decoded_visual_review_pending', 'gpu': gpu, 'cuda_uuid': uuid,
         'videos': videos, 'views': [0,8,16], 'frames': 61, 'time': 'frame/60; playback frame/25 seconds',
         'point_filtering': False, 'checkpoints': {name: {'path':str(path),'sha256':sha(path)} for name,path in paths},
         'scope': 'Same15k schedule, changing only initial duration. Old5k is a different schedule reference.'})
    print(json.dumps({'state':'rendered_decoded_visual_review_pending','lower_region':summary}), flush=True)

if __name__ == '__main__':
    main()
