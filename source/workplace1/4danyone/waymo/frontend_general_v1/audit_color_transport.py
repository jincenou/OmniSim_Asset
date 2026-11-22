"""Check the actual source encoder/normalizer and measure final lossy encoding."""
import hashlib
import json
import sys
from pathlib import Path
from fractions import Fraction

import av
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent
W = ROOT.parent
REPO = W.parent / "repo"
sys.path[:0] = [str(REPO), str(W / "scripts")]
from prepare import encode
from fdanyone.skeleton.pipeline import _video_tensor
from fdanyone.video import write_video, read_rgb_video


def main():
    torch.set_num_threads(2)
    out = ROOT / "color_transport"
    out.mkdir(exist_ok=True)
    assert not (out / "report.json").exists(), "Do not overwrite a completed audit"
    # Gray ramp, saturated primaries, true dark clothing colors, and soft edges.
    palette = np.array([[0, 0, 0], [8, 9, 12], [20, 30, 25], [64, 64, 64],
                        [128, 128, 128], [255, 255, 255], [255, 0, 0],
                        [0, 255, 0], [0, 0, 255], [200, 120, 80]], dtype=np.uint8)
    card = np.repeat(np.repeat(palette[None], 24, axis=0), 48, axis=1)
    ramp = np.broadcast_to(np.rint(np.linspace(0, 255, 480))[None, :, None], (24, 480, 3)).astype(np.uint8)
    rgb = np.concatenate([card, ramp, card, ramp], axis=0)
    alpha = np.broadcast_to(np.linspace(0, 1, 480)[None, :, None], rgb.shape[:2] + (1,))
    composed = np.rint(rgb * alpha + 255 * (1 - alpha)).clip(0, 255).astype(np.uint8)
    premult_twice = np.rint(rgb * alpha * alpha + 255 * (1 - alpha)).clip(0, 255).astype(np.uint8)
    frames = (rgb, composed, np.flip(rgb, axis=1).copy())
    encode(out / "source_rgb_lossless.mp4", [Image.fromarray(x) for x in frames], 25)
    decoded = np.stack(read_rgb_video(out / "source_rgb_lossless.mp4", expected_frames=3))
    original = np.stack(frames)
    max_rgb = int(np.abs(decoded.astype(int) - original).max())
    assert max_rgb <= 1
    Image.fromarray(composed).save(out / "straight_alpha.png")
    png_error = int(np.abs(np.asarray(Image.open(out / "straight_alpha.png")).astype(int) - composed).max())
    assert png_error == 0
    tensor = _video_tensor(out / "source_rgb_lossless.mp4", 3)
    recovered = (tensor[0].permute(1, 2, 3, 0).numpy() + 1) * 127.5
    tensor_error = float(np.abs(recovered - original.astype(float)).max())
    assert tensor_error <= 1
    assert np.all(composed.astype(int) >= rgb.astype(int))
    # A negative control must be detected; it is never used as a model input.
    bad_vs_expected = int(np.abs(premult_twice.astype(int) - composed).max())
    assert bad_vs_expected > 1
    write_video(frames, out / "target_yuv420.mp4", fps=Fraction(25, 1))
    lossy = np.stack(read_rgb_video(out / "target_yuv420.mp4", expected_frames=3))
    delta = lossy.astype(float) - original
    evidence = sorted((W / "frontend_quality_20261020/motion_unmasked").glob("*/input_verification.json"))
    observed = [dict(sample=p.parent.name, **json.loads(p.read_text())) for p in evidence]
    files = [REPO / "fdanyone/video.py", REPO / "fdanyone/skeleton/pipeline.py",
             W / "scripts/prepare.py", W / "scene20/r20261019/scripts/source_foreground.py"]
    report = dict(
        state="transport_tests_passed_not_generation_quality_acceptance",
        source_rgb_roundtrip_max_rgb8=max_rgb, png_roundtrip_max_rgb8=png_error,
        tensor_roundtrip_max_rgb8=tensor_error,
        white_straight_composite_never_darker=True,
        double_alpha_negative_control_max_difference_rgb8=bad_vs_expected,
        final_encoder=dict(codec="libx264", pixel_format="yuv420p", crf=18,
                           rgb_mae=float(np.abs(delta).mean()), rgb_max=float(np.abs(delta).max()),
                           signed_luma_rgb8=float((delta @ np.array([.2126, .7152, .0722])).mean()),
                           scope="Synthetic pattern measured with official writer, not historical target PNG-to-MP4 audit"),
        historical_source_recomposition=observed,
        source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        limits=[
            "Source recomposition uses regenerated RIFE frames; tiny nonzero differences are retained.",
            "White compositing cannot darken pixels but erroneous masks can remove real details.",
            "Target encoding is lossy; no zero-error assertion for YUV420 or the VAE.",
            "Historical pre-encoding target PNGs and RCP outputs were not retained; causal generation-stage localization remains incomplete.",
            "Crop interpolation, generated new-view lighting and model identity fidelity are not certified by transport tests."
        ])
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("historical_source_recomposition", "source_hashes")}))


if __name__ == "__main__":
    main()
