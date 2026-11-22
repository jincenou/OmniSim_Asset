"""Freeze a documented Waymo33k recipe before new heldout quality evaluation."""
import argparse
import math
import runpy
from pathlib import Path
import numpy as np
from grid_common import ROOT, SOURCE, BASE, manifest, save, sha, stamp
from arguments import ModelParams, PipelineParams, OptimizationParams


def make(label, smoke=False):
    parser = argparse.ArgumentParser()
    model, pipe, opt = ModelParams(parser), PipelineParams(parser), OptimizationParams(parser)
    args = parser.parse_args([])
    recipe = SOURCE / 'arguments/neu3d/cook_spinach.py'
    overrides = runpy.run_path(str(recipe))
    for k, v in overrides.items():
        if hasattr(args, k):
            setattr(args, k, v)
    data, path = manifest(label)
    extent = np.asarray([data['motion_aabb_min'], data['motion_aabb_max']])
    bound = math.ceil(float(np.abs(extent).max()) * 1.1 * 20) / 20
    args.grid_args = dict(args.grid_args, bound=bound)
    args.source_path = str(path.parent)
    args.model_path = ''
    args.eval, args.white_background = True, False
    args.load2gpu_on_the_fly = False
    args.iterations = 33000
    args.position_lr_max_steps = args.deform_lr_max_steps = 33000
    args.warm_up = 3000
    dataset = vars(model.extract(args))
    dataset.update(label=label, background_value=.5)
    config = dict(schema=1, label=label, seed=42, created_at=stamp(),
        recipe='Waymo33k: official Neu3D multi-view architecture with explicit known-camera, gray RGBA, domain bound and33k horizon adaptations',
        dataset=dataset, optimization=vars(opt.extract(args)), pipeline=vars(pipe.extract(args)),
        saving_iterations=[5000, 15000, 33000], testing_iterations=[],
        upstream_commit='3ab7b74482910d22fba7d51ac60c8cae1fe24bf1',
        provenance=dict(official_recipe=str(recipe), official_recipe_sha256=sha(recipe),
            manifest=str(path), manifest_sha256=sha(path),
            initialization=str(path.parent / data['canonical_npz']), initialization_sha256=sha(path.parent / data['canonical_npz']),
            bound_source='Pretraining training-view motion hull AABB already in frozen manifest,1.1 margin then round upward0.05; no trained HUST geometry or heldout score',
            motion_aabb=extent.tolist(), bound=bound),
        invariants=dict(train_views=data['train_views'], heldout_views=data['heldout_views'], frame_count=data['frame_count'],
            resolution=[352,640], input_time='frame/(F-1), then upstream internal mapping only',
            training_background=.5, evaluation_background=.5,
            historical_training_background=1., historical_evaluation_background=.5,
            old_initialization_scope='Grid4D and HUST use identical pretraining t0 cloud; FTGS complete recipe uses additional time hulls',
            same_gpu_budget='not established: old wall timing includes data and output; no invented matched GPU-time snapshot',
            native_spatial_lr_scale=5, nominal_to_actual_updates='snapshot before optimizer:5k=4999,15k=14999,33k=32999',
            warmup_actual_updates=2999, dynamic_actual_updates=30000), smoke=smoke)
    if smoke:
        config['optimization'].update(iterations=120, warm_up=20, densify_from_iter=20,
            densify_until_iter=80, densification_interval=20, opacity_reset_interval=60)
        config['saving_iterations']=[60,120]
        config['recipe']='TECHNICAL SMOKE ONLY:120 iterations,warm20,densification20..80; same full image/clip/hash architecture; not a quality result'
        config['invariants'].update(warmup_actual_updates=19,dynamic_actual_updates=100,
            nominal_to_actual_updates='60 snapshot=59 updates;120 snapshot=119 updates')
    return config


if __name__ == '__main__':
    for label in ['N04','N11']:
        for smoke in [False,True]:
            path=ROOT/'configs'/f'{label}{"_smoke" if smoke else ""}.json'
            assert not path.exists()
            save(path,make(label,smoke))
            print(path,flush=True)
