"""Fixed-capacity2x2 supervision ablation from identical full optimizer states."""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from grid_common import (ROOT, PARAMETERS, GaussianModel, KnownScene, new_deform,
                         raw_render, camera, render, save, sha, verify, rng_state, set_rng)
from train_grid import protect_inputs, atomic_torch
from utils.loss_utils import ssim


def equal(a, b):
    if torch.is_tensor(a): assert torch.equal(a.detach().cpu(), b.detach().cpu())
    elif isinstance(a, np.ndarray): assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    else: assert a == b


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);parser.add_argument('--arm',required=True)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    device=verify();sys.addaudithook(protect_inputs)
    plan=json.loads((ROOT/'optimization/PLAN.json').read_text());arm=plan['arms'][args.arm]
    assert args.label in plan['labels']
    source=ROOT/'runs'/f'{args.label}_33k_v1';run=ROOT/'optimization/runs'/f'{args.label}_{args.arm}'
    config=json.loads((source/'config.json').read_text());budget=plan['actual_extra_updates']
    if args.resume:
        assert not (run/'complete.json').exists()
        saved_config=json.loads((run/'config.json').read_text());assert saved_config['optimization_plan_sha256']==sha(ROOT/'optimization/PLAN.json')
        config=saved_config
    else:
        run.mkdir(parents=True,exist_ok=False)
        config.update(continuation_arm=args.arm,optimization_plan_sha256=sha(ROOT/'optimization/PLAN.json'),source_recovery_sha256=sha(source/'recovery.pt'),source_recovery=str(source/'recovery.pt'),extra_update_budget=budget)
        save(run/'config.json',config)
    opt=SimpleNamespace(**config['optimization']);pipe=SimpleNamespace(**config['pipeline'])
    model=GaussianModel(config['dataset']['sh_degree'])
    dataset=SimpleNamespace(**dict(config['dataset'],model_path=str(run)))
    scene=KnownScene(dataset,model)
    loaded=[];cores=[];alphas=[]
    cache=ROOT/'optimization/data'/args.label
    prepared=json.loads((cache/'complete.json').read_text());assert prepared['train_views']==scene.data['train_views']
    core_by_view={v:np.load(cache/f'view{v:02d}.npy',mmap_mode='r') for v in scene.data['train_views']}
    for cam,path in zip(scene.getTrainCameras(),scene.loaded_paths):
        alpha=np.asarray(Image.open(path))[...,3].copy()
        alphas.append(torch.from_numpy(alpha).cuda())
        cores.append(torch.from_numpy(core_by_view[cam.source_view][cam.source_frame].copy()).cuda())
        loaded.append(dict(view=cam.source_view,frame=cam.source_frame))
    save(run/'asset_camera_data.json',json.loads((source/'asset_camera_data.json').read_text()))
    state=torch.load(run/'recovery.pt' if args.resume else source/'recovery.pt',map_location='cpu',weights_only=False)
    for name in PARAMETERS: setattr(model,name,torch.nn.Parameter(state['gaussians'][name].cuda().requires_grad_(True)))
    model.training_setup(opt);model.optimizer.load_state_dict(state['gaussian_optimizer'])
    for name in ['max_radii2D','xyz_gradient_accum','denom']:setattr(model,name,state['statistics'][name].cuda())
    model.active_sh_degree=state['active_sh_degree']
    deform=new_deform(config,regularize=True);deform.train_setting(opt)
    deform.grid4d.load_state_dict(state['hash'],strict=True);deform.deform.load_state_dict(state['deform'],strict=True);deform.optimizer.load_state_dict(state['deform_optimizer'])
    for name in PARAMETERS:equal(getattr(model,name),state['gaussians'][name])
    equal(model.optimizer.state_dict(),state['gaussian_optimizer']);equal(deform.optimizer.state_dict(),state['deform_optimizer'])
    equal(deform.grid4d.state_dict(),state['hash']);equal(deform.deform.state_dict(),state['deform'])
    set_rng(state['rng']);equal(rng_state(),state['rng'])
    cameras=scene.getTrainCameras();by_uid={int(c.uid):i for i,c in enumerate(cameras)}
    stack=None if state['stack'] is None else [by_uid[int(uid)] for uid in state['stack']]
    first=state['extra_updates']+1 if args.resume else 1
    exposure=state['continuation_exposure'] if args.resume else {}
    count=len(model.get_xyz);initial_xyz=model.get_xyz.detach().clone()
    initial_opacity=model._opacity.detach().clone();initial_hash=next(deform.grid4d.parameters()).detach().clone()
    save(run/('resume_validation.json' if args.resume else 'restore_validation.json'),dict(state='passed',source=str(run/'recovery.pt' if args.resume else source/'recovery.pt'),gaussian_hash_deform_adam_rng_exact=True,points=count,first_extra_update=first,heldout_train_reads=0))
    del state
    background=torch.full((3,),.5,device='cuda');started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    trace=(run/'trace.jsonl').open('a' if args.resume else 'x',buffering=1)
    try:
        for step in range(first,budget+1):
            if not stack:stack=list(range(len(cameras)))
            index=stack.pop(random.randrange(len(stack)));cam=cameras[index]
            key=f'{cam.source_view}:{cam.source_frame}';exposure[key]=exposure.get(key,0)+1
            virtual=15000+(step-1)*17999/(budget-1)
            model.update_learning_rate(virtual);deform.update_learning_rate(virtual)
            warp=deform.step(model.get_xyz.detach(),cam.fid.unsqueeze(0).expand(count,-1))
            image=render(cam,model,pipe,background,warp['d_xyz'],warp['d_rotation'],warp['d_scaling'])['render']
            delta=(image-cam.original_image).abs().mean(0);whole=delta.mean();roi=cores[index].long()
            areas=torch.bincount(roi.flatten(),minlength=7).float()[1:]
            masses=torch.zeros(7,device='cuda').scatter_add_(0,roi.flatten(),delta.flatten())[1:]
            valid=areas>0;part=(masses[valid]/areas[valid]).mean() if valid.any() else whole
            rgb_l1=.5*(whole+part) if arm['part_balance'] else whole
            reg=warp['reg_spatial'].mean()*opt.lambda_spatial_tv+warp['reg_temporal'].mean()*opt.lambda_temporal_tv
            loss=.8*rgb_l1+.2*(1-ssim(image,cam.original_image))+reg
            alpha_loss=torch.zeros((),device='cuda');alpha_coverage=None
            # Same geometry, proper alpha. Additional backward cost is reported, not hidden.
            if arm['alpha_supervision'] or step%100==0 or step==1:
                white=render(cam,model,pipe,torch.zeros(3,device='cuda'),warp['d_xyz'],warp['d_rotation'],warp['d_scaling'],override_color=torch.ones_like(model.get_xyz))['render']
                alpha=white[0];target=alphas[index].float()/255;fg=target>.5
                alpha_loss=.5*((alpha-target).abs()[fg].mean()+(alpha-target).abs()[~fg].mean())
                alpha_coverage=float(alpha[fg].detach().mean())
                if arm['alpha_supervision']:loss=loss+.1*alpha_loss
            assert bool(torch.isfinite(loss));loss.backward()
            if step==first or step%100==0:
                norms={}
                for name,params in [('gaussian',[getattr(model,n) for n in PARAMETERS]),('hash',deform.grid4d.parameters()),('deform',deform.deform.parameters())]:
                    grads=[p.grad for p in params if p.grad is not None];assert grads and all(torch.isfinite(g).all() for g in grads)
                    norms[name]=sum(float(g.abs().sum()) for g in grads)
                assert all(v>0 for v in norms.values())
            model.optimizer.step();deform.optimizer.step();model.optimizer.zero_grad(set_to_none=True);deform.optimizer.zero_grad(set_to_none=True)
            assert len(model.get_xyz)==count
            record=dict(extra_update=step,total_actual_updates=32999+step,view=cam.source_view,frame=cam.source_frame,loss=float(loss),global_l1=float(whole),part_l1=float(part),alpha_loss=float(alpha_loss),regularizer=float(reg),foreground_alpha=alpha_coverage,points=count,elapsed_seconds=time.monotonic()-started)
            trace.write(json.dumps(record,allow_nan=False)+'\n')
            if step%100==0 or step==first:
                changes=dict(xyz_max_delta=float((model.get_xyz-initial_xyz).abs().max()),opacity_max_delta=float((model._opacity-initial_opacity).abs().max()),hash_max_delta=float((next(deform.grid4d.parameters())-initial_hash).abs().max()))
                assert all(v>0 for v in changes.values()),changes
                save(run/'progress.json',dict(state='training',device=device,pid=os.getpid(),gradient_norms=norms,parameter_changes=changes,peak_allocated_bytes=torch.cuda.max_memory_allocated(),**record))
            if step%1000==0 or step==budget:
                atomic_torch(run/'recovery.pt',dict(extra_updates=step,gaussians={n:getattr(model,n).detach() for n in PARAMETERS},gaussian_optimizer=model.optimizer.state_dict(),statistics={n:getattr(model,n) for n in ['max_radii2D','xyz_gradient_accum','denom']},active_sh_degree=model.active_sh_degree,hash=deform.grid4d.state_dict(),deform=deform.deform.state_dict(),deform_optimizer=deform.optimizer.state_dict(),rng=rng_state(),stack=[int(cameras[i].uid) for i in stack],continuation_exposure=exposure))
            if step in plan['snapshots']:
                nominal=33000+step;destination=run/'point_cloud'/f'iteration_{nominal}'/'point_cloud.ply'
                assert not destination.exists();model.save_ply(str(destination));deform.save_weights(str(run),nominal)
                flags=deform.reg_spatial_able,deform.reg_temporal_able;deform.reg_spatial_able=deform.reg_temporal_able=False
                refs={}
                with torch.no_grad():
                    for view in [1,9,17]:
                        for frame in [0,(scene.data['frame_count']-1)/2,scene.data['frame_count']-1,12.25]:
                            rgb,a=raw_render(model,deform,pipe,camera(scene.data,view,frame));refs[f'v{view}_f{frame:g}']=torch.cat([rgb,a]).cpu().numpy()
                deform.reg_spatial_able,deform.reg_temporal_able=flags
                snapshots=run/'snapshots';snapshots.mkdir(exist_ok=True);np.savez_compressed(snapshots/f'{nominal}_reference.npz',**refs)
                files=[destination,run/'deform'/f'iteration_{nominal}'/'deform.pth']
                save(snapshots/f'{nominal}.json',dict(iteration=nominal,actual_optimizer_updates=32999+step,extra_updates=step,active_sh_degree=model.active_sh_degree,points=count,files={str(f.relative_to(run)):sha(f) for f in files},state='after_actual_extra_optimizer_step',reference_views=[1,9,17]))
        save(run/'complete.json',dict(state='training_complete_independent_replay_and_quality_pending',device=device,extra_updates=budget,total_actual_updates=32999+budget,points=count,elapsed_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),fixed_capacity=True,trace_sha256=sha(run/'trace.jsonl'),continuation_exposure=exposure))
    finally:trace.close()


if __name__=='__main__':main()
