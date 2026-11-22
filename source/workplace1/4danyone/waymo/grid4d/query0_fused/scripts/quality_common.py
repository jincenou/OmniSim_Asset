"""Asset-only full recovery and snapshots, without modifying previous adapters."""
import json,sys,random,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
NEW=Path(__file__).resolve().parents[1];ROOT=NEW.parent;PREVIOUS=ROOT/'optimization_boundary_20261018'
sys.path.insert(0,str(ROOT/'scripts'))
from grid_common import PARAMETERS,GaussianModel,new_deform,camera,raw_render,render,save,sha,verify,rng_state,set_rng,load_asset,manifest,BASE
from optimize_grid import equal
from train_grid import protect_inputs,atomic_torch
sys.path.insert(0,str(PREVIOUS/'scripts'))
from boundary_common import asset_reference
from boundary_loss import old_alpha,extra_alpha,regions
from utils.loss_utils import ssim

def plan():
    assert sha(NEW/'PLAN.json')==(NEW/'plan_hash.txt').read_text().strip()
    return json.loads((NEW/'PLAN.json').read_text())

def restore(source,regularize=True):
    source=Path(source);config=json.loads((source/'config.json').read_text())
    state=torch.load(source/'recovery.pt',map_location='cpu',weights_only=False)
    model=GaussianModel(config['dataset']['sh_degree'])
    for name in PARAMETERS:setattr(model,name,torch.nn.Parameter(state['gaussians'][name].cuda()))
    opt=SimpleNamespace(**config['optimization']);model.training_setup(opt);model.optimizer.load_state_dict(state['gaussian_optimizer'])
    for name in ['max_radii2D','xyz_gradient_accum','denom']:setattr(model,name,state['statistics'][name].cuda())
    model.active_sh_degree=state['active_sh_degree'];deform=new_deform(config,regularize=regularize);deform.train_setting(opt)
    deform.grid4d.load_state_dict(state['hash'],strict=True);deform.deform.load_state_dict(state['deform'],strict=True);deform.optimizer.load_state_dict(state['deform_optimizer'])
    for name in PARAMETERS:equal(getattr(model,name),state['gaussians'][name])
    equal(model.optimizer.state_dict(),state['gaussian_optimizer']);equal(deform.optimizer.state_dict(),state['deform_optimizer'])
    equal(deform.grid4d.state_dict(),state['hash']);equal(deform.deform.state_dict(),state['deform'])
    set_rng(state['rng']);equal(rng_state(),state['rng'])
    return model,deform,SimpleNamespace(**config['pipeline']),config,json.loads((source/'asset_camera_data.json').read_text()),state

def load_train(data,label):
    full,_=manifest(label);cameras=[];alphas=[];paths=[]
    for view in data['train_views']:
        for frame in range(data['frame_count']):
            path=Path(full['source_data'])/'images'/f'cam{view:02d}'/f'{frame:06d}.png'
            with Image.open(path) as image:
                rgba=torch.tensor(np.asarray(image).copy(),device='cuda',dtype=torch.float32).permute(2,0,1)/255
            rgb=rgba[:3]*rgba[3:]+.5*(1-rgba[3:]);cameras.append(camera(data,view,frame,rgb));alphas.append((rgba[3]*255).round().byte());paths.append(str(path))
    return cameras,alphas,paths

def capture(run,nominal,actual,stage_updates,model,deform,pipe,data,stack,counts,source):
    begin=time.monotonic();saved=rng_state();ply=run/'point_cloud'/f'iteration_{nominal}'/'point_cloud.ply';assert not ply.exists()
    model.save_ply(str(ply));deform.save_weights(str(run),nominal)
    refs=asset_reference(model,deform,pipe,data);(run/'snapshots').mkdir(exist_ok=True)
    np.savez_compressed(run/'snapshots'/f'{nominal}_reference.npz',**refs)
    payload=dict(global_actual_updates=actual,stage_update=stage_updates,gaussians={n:getattr(model,n).detach() for n in PARAMETERS},gaussian_optimizer=model.optimizer.state_dict(),statistics={n:getattr(model,n) for n in ['max_radii2D','xyz_gradient_accum','denom']},active_sh_degree=model.active_sh_degree,hash=deform.grid4d.state_dict(),deform=deform.deform.state_dict(),deform_optimizer=deform.optimizer.state_dict(),rng=saved,stack=list(stack),costs=dict(counts),parent=str(source),quality_plan_sha256=sha(NEW/'PLAN.json'))
    path=run/'recoveries'/f'{nominal}.pt';path.parent.mkdir(exist_ok=True);atomic_torch(path,payload)
    nextlink=run/'recovery.next';nextlink.symlink_to(path.relative_to(run));nextlink.replace(run/'recovery.pt')
    files=[ply,run/'deform'/f'iteration_{nominal}'/'deform.pth',path]
    save(run/'snapshots'/f'{nominal}.json',dict(iteration=nominal,actual_optimizer_updates=actual,stage_update=stage_updates,active_sh_degree=model.active_sh_degree,points=len(model.get_xyz),files={str(p.relative_to(run)):sha(p) for p in files},state='after_actual_optimizer_update',reference_views=[1,9,17]))
    equal(rng_state(),saved)
    return time.monotonic()-begin
