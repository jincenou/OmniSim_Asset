"""Minimal inference loader, avoiding the training dataset/spaCy dependency.

Architecture and diffusion argument mapping follow pinned utils/model_util.py.
Only the unused SMPL mesh converter is omitted for hml_vec inference.
"""
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor/CondMDI"))
from model.mdm_unet import MDM_UNET
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from utils.parser_util import DataOptions, DiffusionOptions, ModelOptions, TrainingOptions


def load_runtime(checkpoint, device):
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint.parent / "args.json").read_text())
    defaults = {}
    for schema in (DataOptions, DiffusionOptions, ModelOptions, TrainingOptions):
        defaults.update(dataclasses.asdict(schema()))
    defaults.update(config)
    a = SimpleNamespace(**defaults)
    assert a.dataset == "humanml" and a.abs_3d and a.keyframe_conditioned
    assert a.arch.startswith("unet") and not a.unconstrained
    assert not a.drop_redundant and not a.xz_only and not a.traj_only
    assert not a.use_random_proj
    clip_root = ROOT / "text_motion/clip"
    assert (clip_root / "status.json").exists()
    clip_status = json.loads((clip_root / "status.json").read_text())
    assert clip_status["state"] == "downloaded_official_sha_verified"
    model = MDM_UNET(
        modeltype="", njoints=263, nfeats=1, num_actions=1, translation=True,
        pose_rep="rot6d", glob=True, glob_rot=True, latent_dim=a.latent_dim,
        ff_size=a.ff_size, num_layers=a.layers, num_heads=4, dropout=.1,
        activation="gelu", data_rep="hml_vec", cond_mode="text",
        cond_mask_prob=a.cond_mask_prob, action_emb="tensor", arch=a.arch,
        emb_trans_dec=a.emb_trans_dec, clip_version="ViT-B/32", dataset=a.dataset,
        two_head=False, dim_mults=a.dim_mults, adagn=a.unet_adagn,
        zero=a.unet_zero, unet_out_mult=a.out_mult, tf_out_mult=a.out_mult,
        xz_only=a.xz_only, keyframe_conditioned=a.keyframe_conditioned,
        keyframe_selection_scheme=a.keyframe_selection_scheme,
        zero_keyframe_loss=a.zero_keyframe_loss,
        load_smpl=False, clip_download_root=str(clip_root))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_kind = "model_avg" if "model_avg" in state else ("model" if "model" in state else "flat")
    weights = state if state_kind == "flat" else state[state_kind]
    expected = model.state_dict()
    missing = set(expected) - set(weights)
    assert not set(weights) - set(expected)
    assert all(k.startswith("clip_model.") for k in missing), sorted(missing)
    # CLIP comes from its independently SHA-verified official checkpoint.
    model.load_state_dict({**{k: expected[k] for k in missing}, **weights}, strict=True)
    model.to(device)
    model.eval()
    assert not a.use_ddim
    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(1000, [1000]),
        conf=gd.DiffusionConfig(
            betas=gd.get_named_beta_schedule(a.noise_schedule, 1000, 1.),
            model_mean_type=gd.ModelMeanType.START_X if a.predict_xstart else gd.ModelMeanType.EPSILON,
            model_var_type=gd.ModelVarType.FIXED_SMALL if a.sigma_small else gd.ModelVarType.FIXED_LARGE,
            loss_type=gd.LossType.MSE, rescale_timesteps=False,
            lambda_vel=a.lambda_vel, lambda_rcxyz=a.lambda_rcxyz, lambda_fc=a.lambda_fc,
            clip_range=a.clip_range, train_trajectory_only_xz=a.xz_only,
            use_random_proj=a.use_random_proj, fp16=a.use_fp16, traj_only=a.traj_only,
            abs_3d=a.abs_3d, apply_zero_mask=a.apply_zero_mask,
            traj_extra_weight=a.traj_extra_weight, time_weighted_loss=a.time_weighted_loss,
            train_x0_as_eps=a.train_x0_as_eps))
    with checkpoint.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    audit = dict(checkpoint=str(checkpoint), sha256=digest, state_kind=state_kind,
                 strict_load=True, separate_official_clip_keys=len(missing),
                 unused_smpl_converter_disabled=True,
                 sampling_steps=1000, configuration=defaults)
    return model, diffusion, a, audit
