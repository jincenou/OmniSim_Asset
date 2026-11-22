"""New-process A replay using exported model and camera/time metadata only."""
import argparse
import json
import os
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
import torch
import grid_context
from grid_common import load_asset, raw_render, camera, sha, save, verify


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    args = parser.parse_args()
    D = grid_context.D
    run = D / "runs" / f"{args.label}_A"
    assert run.is_dir() and (run / "complete.json").exists()
    device = verify()
    cfg = json.loads((D / "PREP_PROTOCOL.json").read_text())
    denied = [D / "data", D / "sources", D / "four_dg/data",
              D.parent / "generation", D.parent / "motion"]
    denied += [Path(p) for p in cfg["sources"].values()]

    def guard(event, values):
        if event == "open" and isinstance(values[0], (str, bytes)):
            path = Path(os.fsdecode(values[0])).resolve()
            assert not any(path.is_relative_to(p.resolve()) for p in denied), str(path)

    sys.addaudithook(guard)
    snapshot = json.loads((run / "snapshots/53000.json").read_text())
    for file, expected in snapshot["files"].items():
        assert sha(run / file) == expected
    model, deform, pipe, config = load_asset(run, 53000)
    data = json.loads((run / "asset_camera_data.json").read_text())
    references = np.load(run / "snapshots/53000_reference.npz")
    errors, backgrounds = [], []
    for key in references.files:
        view, frame = key.split("_")
        cam = camera(data, int(view[1:]), float(frame[1:]))
        rgb, alpha = raw_render(model, deform, pipe, cam)
        actual = torch.cat([rgb, alpha]).cpu().numpy()
        error = float(np.abs(actual - references[key]).max())
        assert error < 1e-5
        assert float(alpha.min()) >= -1e-6 and float(alpha.max()) <= 1.000001
        errors.append(error)
        for value in (0.5, 1.0):
            composed, _ = raw_render(model, deform, pipe, cam, value)
            error = float((composed - (rgb + value * (1 - alpha))).abs().max())
            assert error < 1e-5
            backgrounds.append(error)
    video = run / "replay.mp4"
    assert not video.exists()
    foreground = []
    with imageio.get_writer(str(video), fps=25, codec="libx264", quality=8,
                            macro_block_size=1, ffmpeg_params=["-threads", "2"]) as writer:
        for frame in range(data["frame_count"]):
            rgb, alpha = raw_render(model, deform, pipe, camera(data, 0, frame))
            image = (rgb + 0.5 * (1 - alpha)).clamp(0, 1)
            assert bool(torch.isfinite(image).all())
            foreground.append(int((alpha > 0.5).sum()))
            writer.append_data((image.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
    with imageio.get_reader(str(video)) as reader:
        count = sum(1 for _ in reader)
    assert count == data["frame_count"] and min(foreground) > 0
    save(run / "validation/final_replay.json", {
        "state": "independent_dynamic_replay_passed", "device": device,
        "reference_max_abs_error": max(errors),
        "background_composition_max_abs_error": max(backgrounds),
        "reference_count": len(errors), "noninteger_time_in_references": 12.25,
        "full_video_frames_decoded": count, "minimum_foreground_pixels": min(foreground),
        "video_sha256": sha(video), "source_reads": 0,
        "denied_source_roots": [str(p) for p in denied],
        "quality_approved": False,
        "limitation": "Dynamic replay consistency only; no real hidden-motion, metric-scale or arbitrary-action claim.",
    })


if __name__ == "__main__":
    main()
