"""C1: four single-time fits using the same Gaussian core and native rasterizer."""
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
from grid_common import ROOT,BASE,GaussianModel,BasicPointCloud,camera,manifest,render,save,sha,verify
from train_grid import protect_inputs,atomic_torch
from utils.loss_utils import l1_loss,ssim


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);parser.add_argument('--frame',required=True,type=int)
    args=parser.parse_args();device=verify();sys.addaudithook(protect_inputs)
    plan=json.loads((ROOT/'mechanism/static/PLAN.json').read_text());assert [args.label,args.frame] in plan['cases']
    run=ROOT/'mechanism/static/runs'/f'{args.label}_t{args.frame:03d}';run.mkdir(parents=True,exist_ok=False)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    data,source=manifest(args.label);mapping=np.asarray(data['source_to_normalized'])
    init=BASE/'samples'/args.label/'data/init.npz'
    assert sha(init)==data['source_init_sha256']
    with np.load(init) as z:
        selected=np.isclose(z['times'],args.frame/data['time_denominator'],rtol=0,atol=1e-7)
        assert selected.any()
        positions=(z['positions'][selected]@mapping[:3,:3].T+mapping[:3,3]).astype(np.float32)
        colors=z['colors'][selected].copy()
        assert np.isfinite(colors).all() and colors.min()>=0 and colors.max()<=255
        # Frozen export_4dgs.py uses float byte-valued source colors /255, once.
        colors=np.clip(colors/255.,0,1)
    assert np.isfinite(positions).all() and colors.min()>=0 and colors.max()<=1
    if args.frame==0:
        with np.load(source.parent/data['canonical_npz']) as z:
            assert np.max(np.abs(z['points']-positions))<2e-6 and np.array_equal(z['colors'],colors)
    np.savez_compressed(run/'initialization.npz',points=positions,colors=colors)
    original=json.loads((ROOT/'configs'/f'{args.label}.json').read_text())
    opt=SimpleNamespace(**dict(original['optimization'],iterations=3000,warm_up=3001,
        position_lr_max_steps=3000,densify_from_iter=100,densify_until_iter=1500,densification_interval=100,opacity_reset_interval=1000))
    pipe=SimpleNamespace(**original['pipeline'])
    centers=np.asarray([c['camtoworld'] for c in data['cameras'] if c['view'] in data['train_views']])[:,:3,3]
    extent=float(np.linalg.norm(centers-centers.mean(0),axis=1).max()*1.1)
    model=GaussianModel(3);model.create_from_pcd(BasicPointCloud(points=positions,colors=colors,normals=np.zeros_like(positions)),extent);model.training_setup(opt)
    cameras=[];loaded=[]
    for view in data['train_views']:
        path=Path(data['source_data'])/'images'/f'cam{view:02d}'/f'{args.frame:06d}.png'
        rgba=torch.from_numpy(np.asarray(Image.open(path)).copy()).cuda().float().permute(2,0,1)/255
        cameras.append(camera(data,view,args.frame,rgba[:3]*rgba[3:]+.5*(1-rgba[3:])))
        loaded.append(dict(view=view,path=str(path),sha256=sha(path)))
    save(run/'config.json',dict(label=args.label,frame=args.frame,device=device,optimization=vars(opt),pipeline=vars(pipe),seed=42,
        mode='single_time_static_diagnostic_not_dynamic_asset',background=.5,actual_update_budget=3000,
        initialization_source=str(init),initialization_sha256=sha(init),initialization_frame=args.frame,
        source_to_normalized=data['source_to_normalized'],initialization_difference='frame-specific pretraining21-view visual hull; dynamic baselines start from t0 only',
        reference_scope='Frozen generated multi-view supervision; not real Waymo novel-view ground truth'))
    save(run/'camera_data.json',{k:data[k] for k in ['cameras','width','height','frame_count','time_denominator','train_views','heldout_views']})
    save(run/'training_images.json',loaded)
    background=torch.full((3,),.5,device='cuda');stack=[];started=time.monotonic();exposure={v:0 for v in data['train_views']}
    torch.cuda.reset_peak_memory_stats()
    with (run/'trace.jsonl').open('x',buffering=1) as trace:
        for update in range(1,3001):
            if update%1000==0:model.oneupSHdegree()
            if not stack:stack=cameras.copy()
            cam=stack.pop(random.randrange(len(stack)));exposure[cam.source_view]+=1
            package=render(cam,model,pipe,background,0.,0.,0.)
            image=package['render'];loss=.8*l1_loss(image,cam.original_image)+.2*(1-ssim(image,cam.original_image))
            assert torch.isfinite(loss);loss.backward()
            with torch.no_grad():
                visible=package['visibility_filter'];model.max_radii2D[visible]=torch.maximum(model.max_radii2D[visible],package['radii'][visible])
                if update<opt.densify_until_iter:
                    model.add_densification_stats(package['viewspace_points'],visible)
                    if update>opt.densify_from_iter and update%opt.densification_interval==0:
                        model.densify_and_prune(opt.densify_grad_threshold,.005,extent,20 if update>opt.opacity_reset_interval else None,opt.disable_ws_prune)
                    if update%opt.opacity_reset_interval==0:model.reset_opacity()
                model.optimizer.step();model.update_learning_rate(update);model.optimizer.zero_grad(set_to_none=True)
            record=dict(update=update,view=cam.source_view,loss=float(loss),points=len(model.get_xyz),elapsed_seconds=time.monotonic()-started)
            trace.write(json.dumps(record)+'\n')
            if update%100==0:save(run/'progress.json',dict(state='training',pid=os.getpid(),device=device,**record))
    model.save_ply(str(run/'point_cloud.ply'))
    with torch.no_grad():
        refs={}
        for cam in cameras[::7]:refs[str(cam.source_view)]=render(cam,model,pipe,background,0.,0.,0.)['render'].cpu().numpy()
        np.savez_compressed(run/'reference_render.npz',**refs)
    atomic_torch(run/'optimizer_final.pt',model.optimizer.state_dict())
    save(run/'complete.json',dict(state='static_fit_complete_evaluation_pending',updates=3000,elapsed_seconds=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),points=len(model.get_xyz),active_sh_degree=model.active_sh_degree,
        exposure=exposure,device=device,ply_sha256=sha(run/'point_cloud.ply')))


if __name__=='__main__':main()
