"""Three paired fixed-capacity continuations from complete alpha optimizer states."""
import argparse
import fcntl
import json
import os
import random
import shutil
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from boundary_common import *
from utils.loss_utils import ssim


def load_images(label,data,run):
    item=plan()['identities'][label];rows=json.loads(Path(item['data']['training']).read_text());cache=NEW/'data'/label/'bands'
    complete=json.loads((cache/'complete.json').read_text());assert complete['state']=='passed' and complete['plan_sha256']==sha(NEW/'PLAN.json')
    by_view={v:np.load(cache/f'view{v:02d}.npy',mmap_mode='r') for v in data['train_views']}
    for path,digest in complete['files'].items():assert sha(path)==digest
    cameras=[];alphas=[];partitions=[]
    for row,evidence in zip(rows,complete['images']):
        assert row['path']==evidence['path'] and sha(row['path'])==evidence['sha256']
        with Image.open(row['path']) as image:rgba=torch.from_numpy(np.asarray(image).copy()).cuda()
        f=rgba.float().permute(2,0,1)/255;rgb=f[:3]*f[3:]+.5*(1-f[3:])
        cameras.append(camera(data,row['view'],row['frame'],rgb));alphas.append(rgba[...,3].contiguous())
        partitions.append(torch.from_numpy(by_view[row['view']][row['frame']].copy()).cuda())
    save(run/'training_image_paths.json',[r['path'] for r in rows]);assert len(cameras)==21*data['frame_count']
    return cameras,alphas,partitions


def snapshot(run,step,model,deform,pipe,data,state,stack,exposure,costs,sample_digest,actual_samples):
    nominal=48000+step;begin=time.monotonic();rng=rng_state()
    destination=run/'point_cloud'/f'iteration_{nominal}'/'point_cloud.ply';assert not destination.exists()
    model.save_ply(str(destination));deform.save_weights(str(run),nominal)
    refs=asset_reference(model,deform,pipe,data);(run/'snapshots').mkdir(exist_ok=True);np.savez_compressed(run/'snapshots'/f'{nominal}_reference.npz',**refs)
    recovery=run/'recoveries'/f'{nominal}.pt';recovery.parent.mkdir(exist_ok=True)
    payload=dict(stage_update=step,global_actual_updates=47999+step,gaussians={n:getattr(model,n).detach() for n in PARAMETERS},gaussian_optimizer=model.optimizer.state_dict(),statistics={n:getattr(model,n) for n in ['max_radii2D','xyz_gradient_accum','denom']},active_sh_degree=model.active_sh_degree,hash=deform.grid4d.state_dict(),deform=deform.deform.state_dict(),deform_optimizer=deform.optimizer.state_dict(),rng=rng,stack=list(stack),continuation_exposure=exposure,costs=dict(costs),plan_sha256=sha(NEW/'PLAN.json'),sampling_sha256=sample_digest,actual_samples=actual_samples)
    atomic_torch(recovery,payload)
    link=run/'recovery.next';link.symlink_to(recovery.relative_to(run));link.replace(run/'recovery.pt')
    files=[destination,run/'deform'/f'iteration_{nominal}'/'deform.pth',recovery]
    save(run/'snapshots'/f'{nominal}.json',dict(iteration=nominal,stage_update=step,actual_optimizer_updates=47999+step,active_sh_degree=model.active_sh_degree,points=len(model.get_xyz),files={str(f.relative_to(run)):sha(f) for f in files},state='after_actual_optimizer_update',reference_views=[1,9,17],sampling_sha256=sample_digest,plan_sha256=sha(NEW/'PLAN.json')))
    equal(rng_state(),rng)
    return time.monotonic()-begin


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True,choices=['N04','N11']);parser.add_argument('--arm',required=True,choices=['A','U','B']);parser.add_argument('--trial',default='primary',choices=['primary','repeat']);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    start=time.monotonic();device=verify();sys.addaudithook(protect_inputs);spec=plan();item=spec['identities'][args.label]
    assert json.loads((NEW/'gradient_checks.json').read_text())['state']=='passed'
    if args.trial=='repeat':assert json.loads((NEW/'gate_results.json').read_text())['repeat_authorized'] is True
    run=NEW/'runs'/args.trial/f'{args.label}_{args.arm}';run.mkdir(parents=True,exist_ok=args.resume)
    with (run/'task.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (run/'complete.json').exists()
        model,deform,pipe,config,data,state=restore(args.label,run/'recovery.pt' if args.resume else None,regularize=True)
        if args.resume:
            existing=json.loads((run/'config.json').read_text());assert existing['boundary_plan_sha256']==sha(NEW/'PLAN.json') and state['plan_sha256']==sha(NEW/'PLAN.json');config=existing
        else:
            config.update(boundary_arm=args.arm,boundary_trial=args.trial,boundary_plan_sha256=sha(NEW/'PLAN.json'),source_recovery=str(Path(item['start_run'])/'recovery.pt'),source_recovery_sha256=item['files']['recovery.pt']['sha256'],extra_update_budget=5000)
            save(run/'config.json',config);save(run/'asset_camera_data.json',data)
        cameras,alphas,partitions=load_images(args.label,data,run);by_uid={int(c.uid):i for i,c in enumerate(cameras)};all_uids=list(by_uid)
        protocol=item['sampling'][args.trial];assert sha(protocol['path'])==protocol['sha256'];schedule=json.loads(Path(protocol['path']).read_text())
        assert sha(item['lr']['path'])==item['lr']['sha256'];lrs=json.loads(Path(item['lr']['path']).read_text())
        sampler=torch.load(protocol['state_path'],weights_only=False,map_location='cpu')
        if not args.resume:
            replay=check_start_replay(args.label,model,deform,pipe,data)
            set_rng(state['rng']);first=1;exposure={};stack=list(sampler['start_stack']);actual_samples=[]
            if args.trial=='repeat':
                seed=item['repeat_seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            equal(random.getstate(),sampler['start_rng'])
            costs=dict(training_loop_wall_seconds=0.,snapshot_and_recovery_seconds=0.,logging_seconds=0.,training_render_calls=0,training_backward_calls=0,gaussian_optimizer_calls=0,deform_optimizer_calls=0,added_term_skipped=0)
            save(run/'restore_validation.json',dict(state='passed',device=device,exact_gaussian_hash_deform_adam_restore=True,replay=replay,source=str(Path(item['start_run'])/'recovery.pt'),source_sha256=item['files']['recovery.pt']['sha256'],sampling_sha256=protocol['sha256'],initial_stack_count=len(stack),rng='exact restore' if args.trial=='primary' else 'predeclared reset after exact restore'))
        else:
            set_rng(state['rng']);first=state['stage_update']+1;stack=state['stack'];exposure=state['continuation_exposure'];actual_samples=state['actual_samples'];costs=state['costs']
            equal(random.getstate(),sampler['checkpoints'][first-1]['python_rng']);equal(stack,sampler['checkpoints'][first-1]['stack'])
        del state
        group_state={name:{g['name']:[float(optimizer.state[p]['step']) for p in g['params']] for g in optimizer.param_groups} for name,optimizer in [('gaussian',model.optimizer),('deform',deform.optimizer)]}
        background=torch.full((3,),.5,device='cuda');count=len(model.get_xyz);torch.cuda.reset_peak_memory_stats();initial_xyz=model.get_xyz.detach().clone()
        attempt=int(time.time_ns());trace_path=run/f'trace_{attempt}.jsonl';loop_begin=time.monotonic();opt=SimpleNamespace(**config['optimization'])
        with trace_path.open('x',buffering=1) as trace:
            for step in range(first,5001):
                tick=time.monotonic()
                if not stack:stack=list(all_uids)
                uid=stack.pop(random.randrange(len(stack)));entry=schedule[step-1];assert uid==entry['uid'] and entry['stage_update']==step
                index=by_uid[uid];cam=cameras[index];assert [cam.source_view,cam.source_frame]==[entry['view'],entry['frame']]
                actual_samples.append(uid);key=f'{cam.source_view}:{cam.source_frame}';exposure[key]=exposure.get(key,0)+1
                for optimizer in [model.optimizer,deform.optimizer]:
                    for group in optimizer.param_groups:group['lr']=lrs[step-1][group['name']]
                warp=deform.step(model.get_xyz.detach(),cam.fid.unsqueeze(0).expand(count,-1))
                package=render(cam,model,pipe,background,warp['d_xyz'],warp['d_rotation'],warp['d_scaling']);image=package['render']
                alpha=render(cam,model,pipe,torch.zeros_like(background),warp['d_xyz'],warp['d_rotation'],warp['d_scaling'],override_color=torch.ones_like(model.get_xyz))['render'][0]
                target=alphas[index].float()/255;whole=(image-cam.original_image).abs().mean();dssim=1-ssim(image,cam.original_image)
                rgb=.8*whole+.2*dssim;global_alpha=old_alpha(alpha,target);extra,valid=extra_alpha(alpha,target,partitions[index],args.arm)
                reg=warp['reg_spatial'].mean()*opt.lambda_spatial_tv+warp['reg_temporal'].mean()*opt.lambda_temporal_tv
                loss=rgb+spec['lambda_alpha']*global_alpha+reg+spec['delta']*extra;assert bool(torch.isfinite(loss));loss.backward()
                monitored=step==first or step%100==0;norms={}
                if monitored:
                    for name,params in groups(model,deform).items():
                        gradients=[p.grad for p in params if p.grad is not None];assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
                        norms[name]=float(torch.stack([g.square().sum() for g in gradients]).sum().sqrt())
                model.optimizer.step();deform.optimizer.step();model.optimizer.zero_grad(set_to_none=True);deform.optimizer.zero_grad(set_to_none=True)
                assert len(model.get_xyz)==count;torch.cuda.synchronize();costs['training_loop_wall_seconds']+=time.monotonic()-tick
                costs['training_render_calls']+=2;costs['training_backward_calls']+=1;costs['gaussian_optimizer_calls']+=1;costs['deform_optimizer_calls']+=1;costs['added_term_skipped']+=int(args.arm!='A' and not valid)
                tick=time.monotonic();record=dict(stage_update=step,global_actual_updates=47999+step,view=cam.source_view,frame=cam.source_frame,uid=uid,loss=float(loss),rgb_l1=float(whole),rgb_dssim=float(dssim),rgb_loss=float(rgb),old_alpha_l1=float(global_alpha),added_alpha_l1=float(extra),added_term_valid=valid,regularizer=float(reg),lr={g['name']:g['lr'] for optimizer in [model.optimizer,deform.optimizer] for g in optimizer.param_groups},points=count,loop_elapsed_seconds=time.monotonic()-loop_begin)
                trace.write(json.dumps(record,allow_nan=False)+'\n')
                if monitored:
                    assert shutil.disk_usage(NEW).free>35*2**30
                    save(run/'progress.json',dict(state='training',pid=os.getpid(),device=device,gradient_norms=norms,low_opacity_points=int((model.get_opacity<.01).sum()),frustum_nonzero_radius_points=int(package['visibility_filter'].sum()),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),costs=costs,**record))
                costs['logging_seconds']+=time.monotonic()-tick
                if step in [1000,3000,5000]:
                    equal(random.getstate(),sampler['checkpoints'][step]['python_rng']);equal(stack,sampler['checkpoints'][step]['stack'])
                    for name,optimizer in [('gaussian',model.optimizer),('deform',deform.optimizer)]:
                        for group in optimizer.param_groups:
                            actual=[float(optimizer.state[p]['step']) for p in group['params']];expected=[s+step-first+1 for s in group_state[name][group['name']]];assert actual==expected,(name,group['name'],actual,expected)
                    costs['snapshot_and_recovery_seconds']+=snapshot(run,step,model,deform,pipe,data,None,stack,exposure,costs,protocol['sha256'],actual_samples)
        assert actual_samples==[r['uid'] for r in schedule]
        save(run/'actual_samples.json',actual_samples)
        save(run/'complete.json',dict(state='training_complete_independent_replay_and_quality_pending',device=device,stage_updates=5000,total_actual_updates=52999,points=count,loop_with_io_seconds=time.monotonic()-loop_begin,end_to_end_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),costs=costs,sampling_sha256=protocol['sha256'],actual_samples_sha256=sha(run/'actual_samples.json'),sampling_exact_match=True,plan_sha256=sha(NEW/'PLAN.json'),trace_files={str(p.name):sha(p) for p in run.glob('trace_*.jsonl')}))
        print(args.label,args.arm,args.trial,'5000 actual updates complete',flush=True)


if __name__=='__main__':main()
