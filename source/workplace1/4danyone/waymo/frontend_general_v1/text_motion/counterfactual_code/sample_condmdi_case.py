"""Matched-budget prompt/no-prompt proposals; no hidden-reference evaluation."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from condmdi_runtime import ROOT, load_runtime
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.humanml.scripts.motion_process import recover_from_ric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("branch", choices=["no_text", "text"])
    args = parser.parse_args()
    protocol_path = ROOT / "text_motion/PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text())
    assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == protocol_path.with_suffix(".sha256").read_text().strip()
    assert args.case in protocol["cases"]
    torch.set_num_threads(2)
    assert torch.cuda.device_count() == 1
    gpu = str(torch.cuda.get_device_properties(0).uuid)
    assert gpu == os.environ["FD_EXPECTED_GPU_UUID"].removeprefix("GPU-")
    case = ROOT / "text_motion/cases" / args.case
    data = np.load(case / "condition.npz")
    out = case / args.branch
    out.mkdir(exist_ok=False)
    length = len(data["condition"])
    condition = np.zeros((196, 263), dtype=np.float32)
    mask = np.zeros_like(condition, dtype=bool)
    condition[:length] = data["condition"]
    mask[:length] = data["observed_mask"]
    assert np.all(condition[~mask] == 0)
    x0 = torch.tensor(condition.T[None, :, None], device="cuda")
    observed_mask = torch.tensor(mask.T[None, :, None], device="cuda")
    begin = time.time()
    model, diffusion, config, audit = load_runtime(ROOT / "text_motion/weights/model000750000.pt", "cuda")
    guided = ClassifierFreeSampleModel(model)
    text = protocol["prompts"][args.case.split("_")[0]]
    text_scale = protocol["text_guidance"] if args.branch == "text" else 0.
    # Both branches use the identical two-forward CFG implementation and random stream.
    torch.manual_seed(protocol["seed"])
    noise = torch.randn_like(x0)
    noise_hash = hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()
    y = dict(text=[text], text_scale=torch.tensor([text_scale], device="cuda"),
             lengths=torch.tensor([length], device="cuda"),
             mask=(torch.arange(196, device="cuda") < length)[None, None, None],
             diffusion_steps=1000, imputate=1, stop_imputation_at=0,
             replacement_distribution="conditional", inpainted_motion=x0,
             inpainting_mask=observed_mask, reconstruction_guidance=False)
    with torch.inference_mode():
        output = diffusion.p_sample_loop(guided, x0.shape, noise=noise, device="cuda",
                                         model_kwargs=dict(y=y, obs_x0=x0, obs_mask=observed_mask),
                                         clip_denoised=False, progress=True)
    assert torch.isfinite(output).all()
    observed_error = float((output[observed_mask] - x0[observed_mask]).abs().max())
    assert observed_error <= 1e-5
    raw = output[0, :, 0, :length].T.cpu().numpy() * data["std"] + data["mean"]
    joints = recover_from_ric(torch.tensor(raw, dtype=torch.float32), 22, abs_3d=True).numpy()
    assert np.isfinite(joints).all()
    np.savez_compressed(out / "proposal.npz", features=raw, joints=joints, timestamps=data["time_20fps"])
    report = dict(state="sampled_needs_subject_retargeting_and_observation_alignment",
                  case=args.case, branch=args.branch, gpu_uuid=gpu, seed=protocol["seed"],
                  sampling_steps=1000, forward_evaluations=2000, text=text, text_scale=text_scale,
                  initial_noise_sha256=noise_hash, condition_sha256=hashlib.sha256((case / "condition.npz").read_bytes()).hexdigest(),
                  protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                  weights_sha256=audit["sha256"], checkpoint_state=audit["state_kind"], strict_loaded=True,
                  estimated_observed_feature_max_error=observed_error,
                  elapsed_seconds=time.time() - begin, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  quality_approved=False,
                  limits="HumanML3D-normalized motion proposal, not an SMPL-X asset, source-camera fit, or 4DAnyone video.")
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
