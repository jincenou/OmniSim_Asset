"""Read-only stage evidence around the proven single-GPU Turbo adapter."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import sys
from PIL import Image
import numpy as np

G = Path(__file__).resolve().parent
ROOT = Path(os.environ["FDANYONE_WAYMO_ROOT"])
sys.path.insert(0, os.environ["FDANYONE_RUNTIME"])


def digest_tensor(tensor):
    return hashlib.sha256(tensor.contiguous().view(__import__("torch").uint8).numpy().tobytes()).hexdigest()


def main():
    module = importlib.import_module("fdanyone.model.inference")
    from fdanyone.model.vae import VaeExecutor
    from fdanyone.video import write_video
    from fdanyone.config import INFERENCE
    evidence = ROOT / "stage_evidence"
    evidence.mkdir(exist_ok=False)
    captured = {}
    records = {"target": {}, "rcp": {}}
    original_rcp = module._denoise_rcp
    original_decode = VaeExecutor._decode_and_publish
    original_publish = VaeExecutor.publish_targets

    def capture_rcp(**kwargs):
        result = original_rcp(**kwargs)
        captured["latents"] = result.detach().cpu().clone()
        captured["camera_ids"] = list(kwargs["camera_ids"])
        captured["sha256"] = digest_tensor(captured["latents"])
        return result

    def record_frames(stage, index, frames):
        entries = []
        directory = evidence / stage
        directory.mkdir(exist_ok=True)
        for t in (0, len(frames) // 2, len(frames) - 1):
            frame = frames[t]
            assert frame.dtype == np.uint8 and frame.shape == (1280, 704, 3)
            path = None
            if stage == "rcp" or index in (0, 6, 12, 18):
                path = directory / f"v{index:02d}_t{t:03d}.png"
                Image.fromarray(frame).save(path)
            entries.append(dict(frame=t, rgb_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
                                mean_rgb=frame.mean((0, 1)).tolist(),
                                png=None if path is None else str(path.relative_to(ROOT))))
        records[stage][index] = dict(frames=len(frames), keyframes=entries)

    def decode(self, latents, sink):
        def recording_sink(index, frames):
            record_frames("target", index, frames)
            return sink(index, frames)
        return original_decode(self, latents, recording_sink)

    def publish(self, latents, output_dir, fps):
        # Original target publication first: RCP diagnostic decoding cannot affect target denoising.
        result = original_publish(self, latents, output_dir, fps)
        assert "latents" in captured
        def rcp_sink(index, frames):
            record_frames("rcp", index, frames)
            return write_video(iter(frames), evidence / "rcp" / f"{index:02d}.mp4", fps,
                               crf=INFERENCE.target_h264_crf, preset=INFERENCE.h264_preset)
        original_decode(self, captured["latents"], rcp_sink)
        report = dict(rcp_camera_ids=captured["camera_ids"], rcp_latents_sha256=captured["sha256"],
                      stages=records, target_denoising_unchanged=True,
                      scope="Actual RCP and target RGB before H264; whole-frame RGB summaries are not appearance accuracy")
        (evidence / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return result
    module._denoise_rcp = capture_rcp
    VaeExecutor._decode_and_publish = decode
    VaeExecutor.publish_targets = publish
    sys.argv = ["generate_single.py", "turbo"]
    runpy.run_path(str(G.parent / "scene20/r20261019/scripts/generate_single.py"), run_name="__main__")


if __name__ == "__main__":
    main()
