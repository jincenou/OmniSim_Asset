"""Reproduce SAM CPU-offload transfers on unchanged generated frames.

This probe never writes dataset masks or changes the shared SAM implementation.
"""
import argparse
import gc
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image
import torch

G = Path(__file__).resolve().parent
W = G.parent
D = G / "validation12/downstream"
sys.path[:0] = [str(W / "segmentation"), str(W.parent / "repo")]
from sam2.build_sam import build_sam2_video_predictor
from fdanyone.foreground import predict_foreground_masks
from fdanyone.assets import resolve_foreground_model


def stats(value):
    a = value.detach().float().cpu().numpy()
    return {"shape": list(a.shape), "finite": int(np.isfinite(a).sum()),
            "elements": int(a.size), "min": float(a.min()), "max": float(a.max()),
            "positive": int((a > 0).sum())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--label", default="P09_G0")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--full-sequence", action="store_true")
    a = p.parse_args()
    assert not a.output.exists()
    torch.set_num_threads(2)
    device = subprocess.check_output(["nvidia-smi", f"--id={a.gpu}",
        "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"], text=True).strip().split(", ")
    assert int(device[1]) >= 12000
    assert torch.cuda.device_count() == 1
    assert str(torch.cuda.get_device_properties(0).uuid) == device[0].removeprefix("GPU-")
    temp = D / "data" / a.label / "sam2_input/cam00"
    assert temp.is_dir()
    source = D / "sources" / a.label / "videos/00.mp4"
    frames = []
    with av.open(str(source)) as c:
        c.streams.video[0].thread_count = 1
        for i, f in enumerate(c.decode(video=0)):
            if i % 15 == 0:
                frames.append(np.array(f.to_image().resize((352, 640), Image.Resampling.LANCZOS)))
    predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml",
        str(W / "segmentation/sam2.1_hiera_large.pt"), device="cuda", apply_postprocessing=False)
    masks = predict_foreground_masks(tuple(frames), resolve_foreground_model(W.parent / "repo/models"), "cuda:0")
    seed = masks[0] > 127
    assert seed.sum() >= 100
    original = predictor._run_single_frame_inference
    original_memory = predictor._run_memory_encoder
    rows = []
    # Original first, then explicit transfer synchronization and device-resident state.
    for mode in ("original", "synchronized_offload", "gpu_state"):
        transfers = []
        if mode == "synchronized_offload":
            if a.full_sequence:
                from sam_transfer_guard import install_transfer_guard
                install_transfer_guard(predictor)
            def synced(*args, **kwargs):
                result = original(*args, **kwargs)
                cpu_mask = result[0]["pred_masks"]
                # Snapshot CPU contents before making the asynchronous copy safe.
                before = cpu_mask.clone() if cpu_mask.device.type == "cpu" else None
                torch.cuda.current_stream().synchronize()
                if before is not None:
                    after = cpu_mask.clone()
                    expected = result[1].detach().cpu()
                    transfers.append({
                        "cpu_changed_after_sync": int((before != after).sum()),
                        "pre_sync_error": float((before - expected).abs().max()),
                        "post_sync_error": float((after - expected).abs().max()),
                    })
                return result
            def synced_memory(*args, **kwargs):
                result = original_memory(*args, **kwargs)
                torch.cuda.current_stream().synchronize()
                return result
            if not a.full_sequence:
                predictor._run_single_frame_inference = synced
                predictor._run_memory_encoder = synced_memory
        else:
            predictor._run_single_frame_inference = original
            predictor._run_memory_encoder = original_memory
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path=str(temp), offload_video_to_cpu=True,
                                         offload_state_to_cpu=mode != "gpu_state")
            _, _, seed_out = predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=seed)
            seed_stats = stats(seed_out)
            outputs = []
            # Original arm stays a 3-frame repro, full arms verify the actual fix.
            maximum = None if a.full_sequence and mode != "original" else 2
            for idx, ids, logits in predictor.propagate_in_video(state, max_frame_num_to_track=maximum):
                assert ids == [1]
                outputs.append(logits.detach().float().cpu().numpy().copy())
            rows.append({"mode": mode, "seed_output": seed_stats,
                         "frames": [stats(torch.from_numpy(x)) for x in outputs],
                         "transfers": transfers})
        if mode == "synchronized_offload":
            safe = outputs
        if mode == "gpu_state":
            rows[-1]["max_diff_from_synchronized_offload"] = [
                float(np.max(np.abs(x - y))) for x, y in zip(outputs, safe, strict=True)]
        print(json.dumps(rows[-1]), flush=True)
        del state, outputs
        gc.collect()
        torch.cuda.empty_cache()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({
        "label": a.label, "gpu": a.gpu, "uuid": device[0], "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "seed_area": int(seed.sum()), "reference_count": len(masks), "rows": rows,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "scope": "Full sequence safe-CPU/GPU parity; original first3 only." if a.full_sequence else
                 "First3 frames only; not full sequence or mask-quality acceptance.",
    }, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
