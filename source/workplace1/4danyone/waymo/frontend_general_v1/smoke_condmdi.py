"""Actual pretrained sampling on the official public example; NOT a Waymo result."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import clip
import numpy as np
import torch

from condmdi_runtime import ROOT, load_runtime
from condmdi_mask import observation_mask, normalized_condition
from data_loaders.humanml.scripts.motion_process import process_file, recover_root_rot_pos, recover_from_ric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    assert torch.cuda.device_count() == 1
    actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
    assert actual_uuid == os.environ["FD_EXPECTED_GPU_UUID"].removeprefix("GPU-")
    out = ROOT / "text_motion/smoke"
    out.mkdir(exist_ok=False)
    repo = ROOT / "vendor/CondMDI"
    os.chdir(repo)
    # Public official example is a deployment smoke test, never inserted in Waymo data.
    original = np.load(repo / "dataset/000021.npy")[:81, :22]
    features, reference, _, _ = process_file(torch.tensor(original, dtype=torch.float32), .002)
    features = torch.tensor(features, dtype=torch.float32)
    _, root, angle = recover_root_rot_pos(features, return_rot_ang=True)
    features[:, 0] = angle
    features[:, 1:4] = root[:, [0, 2, 1]]
    roundtrip = recover_from_ric(features, 22, abs_3d=True).numpy()
    error = float(np.abs(roundtrip - reference[:-1]).max())
    assert error < 1e-5
    support = np.ones((81, 22), dtype=bool)
    support[30:50, [4, 5, 7, 8, 10, 11]] = False
    feature_mask = observation_mask(support, np.ones(81, dtype=bool))
    mean = np.load(repo / "dataset/HumanML3D_abs/Mean_abs_3d.npy")
    std = np.load(repo / "dataset/HumanML3D_abs/Std_abs_3d.npy")
    condition = normalized_condition(features.numpy(), feature_mask, mean, std)
    observed = np.zeros((196, 263), dtype=np.float32)
    mask = np.zeros_like(observed, dtype=bool)
    observed[:80] = condition
    mask[:80] = feature_mask
    x0 = torch.tensor(observed.T[None, :, None], device="cuda")
    obs_mask = torch.tensor(mask.T[None, :, None], device="cuda")
    started = time.time()
    model, diffusion, config, audit = load_runtime(args.checkpoint.resolve(), "cuda")
    torch.manual_seed(42)
    y = dict(text=[""], uncond=True, lengths=torch.tensor([80], device="cuda"),
             mask=(torch.arange(196, device="cuda") < 80)[None, None, None],
             diffusion_steps=1000, imputate=1, stop_imputation_at=0,
             replacement_distribution="conditional", inpainted_motion=x0,
             inpainting_mask=obs_mask, reconstruction_guidance=False)
    with torch.inference_mode():
        noise = torch.randn_like(x0)
        test_t = torch.tensor([500], device="cuda")
        walk = model(noise, test_t, y=dict(text=["A person walks steadily."]), obs_x0=x0, obs_mask=obs_mask)
        stand = model(noise, test_t, y=dict(text=["A person stands still."]), obs_x0=x0, obs_mask=obs_mask)
        text_delta = float((walk - stand).abs().mean())
        assert text_delta > 0 and np.isfinite(text_delta)
        sample = diffusion.p_sample_loop(model, x0.shape, model_kwargs=dict(y=y, obs_x0=x0, obs_mask=obs_mask),
                                        device="cuda", clip_denoised=False, progress=True,
                                        noise=noise)
    assert torch.isfinite(sample).all()
    preserved_error = float((sample[obs_mask] - x0[obs_mask]).abs().max())
    assert preserved_error <= 1e-5
    raw = sample[0, :, 0, :80].T.cpu().numpy() * std + mean
    xyz = recover_from_ric(torch.tensor(raw, dtype=torch.float32), 22, abs_3d=True).numpy()
    assert xyz.shape == (80, 22, 3) and np.isfinite(xyz).all()
    np.savez_compressed(out / "sample.npz", generated=xyz, normalized_output=sample.cpu().numpy(),
                        reference=reference[:80], observed_mask=feature_mask, support=support[:80],
                        timestamps=np.arange(80) / 20.)
    report = dict(state="pretrained_sampling_smoke_passed_not_waymo_validation", **audit,
                  gpu_uuid=actual_uuid, seed=42, valid_frames=80, padded_model_frames=196,
                  actual_sampler_steps=1000, no_text=True, imputation=True,
                  representation_roundtrip_max=error, observed_feature_max_error=preserved_error,
                  text_condition_forward_mean_difference=text_delta,
                  content_tokens={s: int((clip.tokenize(s) != 0).sum()) - 2
                                  for s in ["A person walks steadily.", "A person stands still."]},
                  elapsed_seconds=time.time() - started,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  scope="Official public example, no actual Waymo gap completion or quality conclusion",
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "configuration"}), flush=True)


if __name__ == "__main__":
    main()
