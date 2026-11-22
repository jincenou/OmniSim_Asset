"""Four existing RCP views as flow-matching target conditions; no new references."""
import hashlib
import torch


def tensor_sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def apply_target_anchors(module, original_targets, kwargs, rcp_latents, camera_ids, enabled=True):
    """Keep the official grouping/forward path and constrain only four target rows.

    RCP outputs are generated hypotheses, not observations. This is an experimental
    engineering intervention, not the official inference algorithm.
    """
    if not enabled:
        return original_targets(**kwargs), {"enabled": False}
    initial = kwargs["initial_latents"]
    scheduler = kwargs["denoiser"].scheduler
    routes = kwargs["routes"]
    ids = tuple(camera_ids[:4])
    assert len(camera_ids) == 6 and ids == (4, 9, 14, 19)
    assert initial.shape[0] == 24 and initial.device.type == "cpu"
    assert rcp_latents.shape[0] == 6 and rcp_latents.shape[1:] == initial.shape[1:]
    assert kwargs["src_latents"].shape[0] == 5
    assert torch.equal(kwargs["src_latents"][1:5].cpu(), rcp_latents[:4].cpu())
    assert not scheduler.inverse_timesteps and not scheduler.reverse_sigmas
    sigmas = scheduler.sigmas.detach().cpu().float()
    assert len(sigmas) == len(routes) == len(scheduler.timesteps) == 4
    assert float(sigmas[0]) == 1 and bool((sigmas[:-1] > sigmas[1:]).all())
    # The exact initial target noise is reused; no additional random draws.
    index = torch.tensor(ids, dtype=torch.long)
    noise = initial.index_select(0, index).clone()
    clean = rcp_latents[:4].detach().cpu().clone()
    source_hash = tensor_sha(kwargs["src_latents"])
    initial_hash = tensor_sha(initial)
    planned = [(step, tuple(group)) for step, groups in enumerate(routes) for group in groups]
    for groups in routes:
        assert sorted(v for group in groups for v in group) == list(range(24))
    calls = []
    original_group = module.denoise_group

    def blend(clean_part, noise_part, sigma, target):
        # Same marginal interpolation as native FlowMatchScheduler.add_noise.
        return ((1 - sigma) * clean_part.float() + sigma * noise_part.float()).to(target)

    def conditioned_group(denoiser, latents, source, context, pose, null_pose, step):
        expected_step, views = planned[len(calls)]
        assert step == expected_step and denoiser is kwargs["denoiser"]
        pairs = [(local, ids.index(view)) for local, view in enumerate(views) if view in ids]
        sigma = float(sigmas[step])
        sigma_next = float(sigmas[step + 1]) if step + 1 < len(sigmas) else 0.0
        if pairs:
            local_index = torch.tensor([p[0] for p in pairs], device=latents.device)
            reference_index = torch.tensor([p[1] for p in pairs])
            c, n = clean[reference_index], noise[reference_index]
            latents.index_copy_(0, local_index, blend(c, n, sigma, latents))
        result = original_group(denoiser, latents, source, context, pose, null_pose, step)
        if pairs:
            result.index_copy_(0, local_index, blend(c, n, sigma_next, result))
        calls.append(dict(step=step, views=list(views), anchored_views=[views[p[0]] for p in pairs],
                          sigma=sigma, sigma_next=sigma_next))
        return result

    module.denoise_group = conditioned_group
    try:
        result = original_targets(**kwargs)
    finally:
        module.denoise_group = original_group
    assert len(calls) == len(planned)
    assert torch.equal(result.index_select(0, index), clean.to(result))
    assert tensor_sha(kwargs["src_latents"]) == source_hash
    report = dict(
        enabled=True, policy="existing_four_rcp_target_anchors_v1",
        anchor_camera_ids=list(ids), reference_count=4, source_packing_unchanged=True,
        initial_target_noise_sha256=initial_hash, anchor_noise_sha256=tensor_sha(noise),
        reference_latents_sha256=tensor_sha(clean), source_latents_sha256=source_hash,
        sigmas=sigmas.tolist(), calls=calls, final_anchor_equality=True,
        extra_model_forwards=0, extra_random_draws=0,
        interpretation="Generated RCP references are NOT ground truth; anchor identity is imposed, not a quality metric.",
        quality_approved=False)
    return result, report
