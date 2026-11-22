"""Verified default-versus-native temporal setting metrics and synchronized videos."""
from argparse import ArgumentParser
from contextlib import ExitStack
from datetime import datetime
import importlib.util
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from PIL import Image

A = Path(__file__).resolve().parent
R = A.parents[1]
spec = importlib.util.spec_from_file_location('fd_temporal_video', R / 'backend/src/video_io.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def open_video(path, stack, frames):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) == 25
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == frames
    stack.callback(cap.release)
    return cap


def decode(cap):
    ok, pixels = cap.read()
    assert ok and pixels.shape == (640, 704, 3)
    return cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)


def main():
    parser = ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.run / 'experiment.json').read_text())
    assert config['temporal_control'] in ('grid61', 'smooth001')
    baseline = Path(config['controlled_baseline'])
    before = json.loads((baseline / 'evaluation/results.json').read_text())
    after = json.loads((args.run / 'evaluation/results.json').read_text())
    assert before['crops'] == after['crops']
    assert [(row['view'], row['frame'], row['background'], row['input_sha256']) for row in before['samples']] == [
        (row['view'], row['frame'], row['background'], row['input_sha256']) for row in after['samples']]
    base_profile = json.loads((A / 'diagnostics/quality_profile_v1' / baseline.name / 'results.json').read_text())
    new_profile = json.loads((A / 'diagnostics/quality_profile_v1' / args.run.name / 'results.json').read_text())
    frames = after['frame_count']
    assert before['frame_count'] == frames
    out = args.run / 'temporal_comparison'
    out.mkdir(exist_ok=False)
    video_records = []
    with ExitStack() as stack:
        for view in (0, 8, 16):
            left = open_video(baseline / 'evaluation' / f'view{view:02d}.mp4', stack, frames)
            right = open_video(args.run / 'evaluation' / f'view{view:02d}.mp4', stack, frames)
            path = out / f'view{view:02d}.mp4'
            writer = stack.enter_context(module.MP4Writer(str(path), 25, 1056, 640))
            for frame in range(frames):
                b, a = decode(left), decode(right)
                combined = np.concatenate([b[:, :352], b[:, 352:], a[:, 352:]], axis=1)
                writer.append_data(combined)
                if frame in (0, frames // 2, frames - 1):
                    Image.fromarray(combined).save(out / f'view{view:02d}_frame{frame:03d}.jpg', quality=95)
            assert not left.read()[0] and not right.read()[0]
            video_records.append(path)
    for path in video_records:
        cap = cv2.VideoCapture(str(path))
        count = 0
        while True:
            ok, pixels = cap.read()
            if not ok:
                break
            assert pixels.shape == (640, 1056, 3)
            count += 1
        cap.release()
        assert count == frames
    b, a = before['mean']['0.5'], after['mean']['0.5']
    result = {'updated_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'state': 'native_temporal_metrics_inputs_and_three_full_comparison_videos_verified_visual_review_pending',
        'control': config['temporal_control'], 'baseline': str(baseline), 'run': str(args.run),
        'before_gray': b, 'after_gray': a, 'delta': {key: a[key] - b[key] for key in b},
        'lpips_reduction_percent': 100 * (b['crop_lpips'] - a['crop_lpips']) / b['crop_lpips'],
        'ssim': {'before': base_profile['ssim_mean'], 'after': new_profile['ssim_mean']},
        'dense_lower_region': {'before': base_profile['mean_dense'], 'after': new_profile['mean_dense']},
        'gaussians': [before['gaussians'], after['gaussians']],
        'coarse_steps': 3000, 'fine_steps': 30000, 'frames': frames,
        'scope': 'Same captured trainer,inputs,crops,split,seed,RGB objective and33k budget;sole explicit temporal hidden field differs. Adaptive density and grid capacity may change. Video is generated/reference/default/new; metrics are direct floating model renders,not recompressed video. Generated validation,not real novel-view GT.',
        'production_approved': False}
    (out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'control': result['control'], 'lpips_reduction_percent': result['lpips_reduction_percent']}))


if __name__ == '__main__':
    main()
