"""Exact alpha recovery and shared run helpers; all new artifacts stay in NEW."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

NEW=Path(__file__).resolve().parents[1]
ROOT=NEW.parent
sys.path.insert(0,str(ROOT/'scripts'))
from grid_common import (PARAMETERS, GaussianModel, new_deform, camera, raw_render,
                         render, save, sha, verify, rng_state, set_rng, load_asset)
from optimize_grid import equal
from train_grid import protect_inputs, atomic_torch
from boundary_loss import old_alpha, extra_alpha, regions


def plan():
    assert sha(NEW/'PLAN.json')==(NEW/'plan_hash.txt').read_text().strip()
    return json.loads((NEW/'PLAN.json').read_text())


def restore(label, recovery=None, regularize=True):
    specification=plan();item=specification['identities'][label];source=Path(item['start_run'])
    path=Path(recovery) if recovery else source/'recovery.pt'
    if recovery is None:assert sha(path)==item['files']['recovery.pt']['sha256']
    config=json.loads((source/'config.json').read_text());state=torch.load(path,map_location='cpu',weights_only=False)
    model=GaussianModel(config['dataset']['sh_degree'])
    for name in PARAMETERS:setattr(model,name,torch.nn.Parameter(state['gaussians'][name].cuda().requires_grad_(True)))
    opt=SimpleNamespace(**config['optimization']);model.training_setup(opt);model.optimizer.load_state_dict(state['gaussian_optimizer'])
    for name in ['max_radii2D','xyz_gradient_accum','denom']:setattr(model,name,state['statistics'][name].cuda())
    model.active_sh_degree=state['active_sh_degree']
    deform=new_deform(config,regularize=regularize);deform.train_setting(opt)
    deform.grid4d.load_state_dict(state['hash'],strict=True);deform.deform.load_state_dict(state['deform'],strict=True);deform.optimizer.load_state_dict(state['deform_optimizer'])
    for name in PARAMETERS:equal(getattr(model,name),state['gaussians'][name])
    equal(model.optimizer.state_dict(),state['gaussian_optimizer']);equal(deform.optimizer.state_dict(),state['deform_optimizer'])
    equal(deform.grid4d.state_dict(),state['hash']);equal(deform.deform.state_dict(),state['deform'])
    actual_gaussian=[g['params'][0] for g in model.optimizer.param_groups]
    wanted=[model._xyz,model._features_dc,model._features_rest,model._opacity,model._scaling,model._rotation]
    assert all(a is b for a,b in zip(actual_gaussian,wanted))
    assert [g['name'] for g in model.optimizer.param_groups]==['xyz','f_dc','f_rest','opacity','scaling','rotation']
    assert [g['name'] for g in deform.optimizer.param_groups]==['deform','grid']
    assert all(a is b for a,b in zip(deform.optimizer.param_groups[0]['params'],deform.deform.parameters()))
    assert all(a is b for a,b in zip(deform.optimizer.param_groups[1]['params'],deform.grid4d.parameters()))
    assert len(model.get_xyz)==item['points']
    set_rng(state['rng']);equal(rng_state(),state['rng'])
    data=json.loads((source/'asset_camera_data.json').read_text())
    return model,deform,SimpleNamespace(**config['pipeline']),config,data,state


@torch.no_grad()
def check_start_replay(label,model,deform,pipe,data):
    start=Path(plan()['identities'][label]['start_run']);refs=np.load(start/'snapshots/48000_reference.npz')
    flags=deform.reg_spatial_able,deform.reg_temporal_able;saved=rng_state()
    deform.reg_spatial_able=deform.reg_temporal_able=False;rows=[]
    try:
        for key in refs.files:
            v,f=key.split('_');rgb,alpha=raw_render(model,deform,pipe,camera(data,int(v[1:]),float(f[1:])))
            error=float(np.abs(torch.cat([rgb,alpha]).cpu().numpy()-refs[key]).max());assert error<1e-5,(key,error)
            rows.append(dict(probe=key,max_abs_error=error))
    finally:
        deform.reg_spatial_able,deform.reg_temporal_able=flags;set_rng(saved)
    return rows


def groups(model,deform):
    return dict(geometry=[model._xyz,model._scaling,model._rotation],opacity=[model._opacity],SH=[model._features_dc,model._features_rest],hash=list(deform.grid4d.parameters()),deform=list(deform.deform.parameters()))


def asset_reference(model,deform,pipe,data):
    saved=rng_state();flags=deform.reg_spatial_able,deform.reg_temporal_able
    deform.reg_spatial_able=deform.reg_temporal_able=False;refs={}
    try:
        with torch.no_grad():
            for view in [1,9,17]:
                for frame in [0,(data['frame_count']-1)/2,data['frame_count']-1,12.25]:
                    rgb,a=raw_render(model,deform,pipe,camera(data,view,frame));refs[f'v{view}_f{frame:g}']=torch.cat([rgb,a]).cpu().numpy()
    finally:
        deform.reg_spatial_able,deform.reg_temporal_able=flags;set_rng(saved)
    return refs
