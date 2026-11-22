"""CPU integration test of actual target loop and scheduler with a synthetic denoiser."""
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from reference_anchor import apply_target_anchors

G = Path(__file__).resolve().parent
sys.path.insert(0, str(G.parent.parent / "repo"))
from fdanyone.model import inference as module
from fdanyone.model.routing import routing_steps
from fdanyone.views import ViewPlan

spec = importlib.util.spec_from_file_location(
    "native_scheduler", G.parent.parent / "repo/fdanyone/vendor/diffsynth/schedulers/flow_match.py")
sched_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sched_module)


class Features:
    num_features = 24
    null_features = torch.zeros(1)
    def allocate_group(self, n, device):
        return torch.zeros(n)
    def copy_group(self, views, dest):
        dest.copy_(torch.tensor(views))


class Model:
    def __init__(self):
        self.calls = 0
    def __call__(self, x, x_src, timestep, pose_features, **kwargs):
        self.calls += 1
        # Cross-view coupling makes wrong camera row mapping observable.
        return (x * .2 + x.mean(0, keepdim=True) * .1 + pose_features[:, None, None, None, None] * .01).to(x)


def main():
    torch.set_num_threads(2)
    scheduler = sched_module.FlowMatchScheduler(num_inference_steps=4, shift=5, sigma_min=0, extra_one_step=True)
    # Shifted groups exercise camera-ID mapping independently of the local row.
    routes = tuple(tuple(tuple((offset + j + step * 2) % 24 for j in range(6))
                         for offset in range(0, 24, 6)) for step in range(4))
    rng = torch.Generator().manual_seed(42)
    initial = torch.randn((24, 2, 2, 2, 2), generator=rng)
    rcp = torch.randn((6, 2, 2, 2, 2), generator=rng)
    source = torch.cat([torch.zeros_like(rcp[:1]), rcp[:4]])
    model = Model()
    denoiser = SimpleNamespace(dtype=torch.float32, scheduler=scheduler, model=model)
    kwargs = dict(denoiser=denoiser, src_latents=source, context=torch.zeros(1),
                  pose_features=Features(), routes=routes, device="cpu")
    module._bf16_autocast = contextlib.nullcontext
    original_group = module.denoise_group
    before_rng = torch.random.get_rng_state().clone()
    original = module._denoise_targets_single(initial_latents=initial.clone(), **kwargs)
    disabled, _ = apply_target_anchors(module, module._denoise_targets_single,
        dict(kwargs, initial_latents=initial.clone()), rcp, (4, 9, 14, 19, 0, 12), enabled=False)
    assert torch.equal(original, disabled)
    result, report = apply_target_anchors(module, module._denoise_targets_single,
        dict(kwargs, initial_latents=initial.clone()), rcp, (4, 9, 14, 19, 0, 12))
    assert module.denoise_group is original_group
    assert model.calls == 48
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert torch.equal(result[[4, 9, 14, 19]], rcp[:4])
    assert not torch.equal(result[0], original[0]), "Test must exercise cross-view influence"
    # Native scheduler correspondence, endpoints and BF16 cast contract.
    errors = []
    for step, t in enumerate(scheduler.timesteps):
        sigma = float(scheduler.sigmas[step])
        manual = (1 - sigma) * rcp[:4] + sigma * initial[[4, 9, 14, 19]]
        native = scheduler.add_noise(rcp[:4], initial[[4, 9, 14, 19]], t)
        errors.append(float((manual - native).abs().max()))
    assert max(errors) == 0
    bf_model = Model()
    bf_kwargs = dict(kwargs, denoiser=SimpleNamespace(dtype=torch.bfloat16, scheduler=scheduler, model=bf_model),
                     src_latents=source.bfloat16(), initial_latents=initial.bfloat16())
    bf_result, _ = apply_target_anchors(module, module._denoise_targets_single,
        bf_kwargs, rcp.bfloat16(), (4, 9, 14, 19, 0, 12))
    assert bf_result.dtype == torch.bfloat16
    assert torch.equal(bf_result[[4, 9, 14, 19]], rcp[:4].bfloat16())
    assert bf_model.calls == 16 and torch.equal(before_rng, torch.random.get_rng_state())
    rejected = False
    try:
        apply_target_anchors(module, module._denoise_targets_single,
            dict(kwargs, initial_latents=initial.clone()), rcp, (4, 14, 9, 19, 0, 12))
    except AssertionError:
        rejected = True
    assert rejected and module.denoise_group is original_group
    output = G / "generation_dev6/anchor_test.json"
    output.write_text(json.dumps(dict(
        state="passed", actual_target_loop_and_scheduler=True, neural_model="synthetic",
        null_intervention_bitwise_equal=True, final_four_anchors_exact=True,
        bf16_loop_and_endpoint_passed=True,
        sampler_matches_native_add_noise_max_error=max(errors), rng_unchanged=True,
        invalid_packing_rejected=True, forwards_per_arm=16, report=report), indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
