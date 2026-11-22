"""Run the pinned upstream optimization with explicit data, monitoring and recovery."""
import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from grid_common import (ROOT, BASE, SOURCE, PARAMETERS, KnownScene, camera, raw_render,
                         rng_state, set_rng, sha, save, stamp, verify)
import train as native


class StopRequested(Exception):
    pass


def atomic_torch(path, value):
    temp = path.with_suffix('.writing')
    torch.save(value, temp)
    temp.replace(path)


def protect_inputs(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = Path(os.fsdecode(args[0])).absolute()
        if path.is_relative_to(BASE):
            assert not (args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)), str(path)
            assert not (path.suffix.lower() in ['.png','.jpg','.mp4','.npz'] and
                        any(f'/cam{v:02d}/' in str(path) for v in [0,8,16])), 'Heldout content forbidden in training'


class Monitor:
    def __init__(self, config, run, device, resume, stop_after):
        self.config, self.run, self.device = config, run, device
        self.resume_path, self.stop_after = resume, stop_after
        self.opt = SimpleNamespace(**config['optimization'])
        self.current = 0
        self.counts = dict(optimizer_calls=0, dynamic_gradient_updates=0, training_renders=0,
                           image_pixels_exposed=0, reference_foreground_pixels_exposed=0)
        self.wall_training = 0.
        self.gpu_timeline_ms = 0.
        self.last_sync = time.monotonic()
        self.scene = None
        self.events = []
        self.exposure = {}
        self.trace = (run/'trace.jsonl').open('a' if resume else 'x', buffering=1)
        self.gpu_start = torch.cuda.Event(enable_timing=True)
        self.gpu_end = torch.cuda.Event(enable_timing=True)

    def initialize(self, scene, gs, deform):
        self.scene = scene
        self.deform = deform
        if not self.resume_path:
            self.initial_xyz = gs._xyz.detach().clone()
            self.install_events(gs)
            return 1, None
        state = torch.load(self.resume_path, map_location='cpu')
        assert state['config_sha256'] == sha(self.run/'config.json')
        for name in PARAMETERS:
            setattr(gs, name, torch.nn.Parameter(state['gaussians'][name].cuda().requires_grad_(True)))
        gs.training_setup(self.opt)
        gs.optimizer.load_state_dict(state['gaussian_optimizer'])
        for name in ['max_radii2D','xyz_gradient_accum','denom']:
            setattr(gs,name,state['statistics'][name].cuda())
        gs.active_sh_degree = state['active_sh_degree']
        deform.grid4d.load_state_dict(state['hash'], strict=True)
        deform.deform.load_state_dict(state['deform'], strict=True)
        deform.optimizer.load_state_dict(state['deform_optimizer'])
        self.counts, self.exposure = state['counts'], state['exposure']
        self.wall_training, self.gpu_timeline_ms = state['wall_training'],state['gpu_timeline_ms']
        cameras = {int(c.uid):c for c in scene.getTrainCameras()}
        stack = None if state['stack'] is None else [cameras[u] for u in state['stack']]
        # Keep the generators' byte states on CPU as required by set_rng_state.
        set_rng(state['rng'])
        def equal_state(a,b,path='state'):
            if torch.is_tensor(a):
                assert torch.equal(a.detach().cpu(),b.detach().cpu()),path
            elif isinstance(a,np.ndarray):
                assert np.array_equal(a,b),path
            elif isinstance(a,dict):
                assert a.keys()==b.keys(),path
                for k in a:equal_state(a[k],b[k],path+'/'+str(k))
            elif isinstance(a,(list,tuple)):
                assert len(a)==len(b),path
                for i,(x,y) in enumerate(zip(a,b)):equal_state(x,y,path+'/'+str(i))
            else:
                assert a==b,path
        for name in PARAMETERS:equal_state(getattr(gs,name),state['gaussians'][name],name)
        equal_state(gs.optimizer.state_dict(),state['gaussian_optimizer'],'gaussian_optimizer')
        equal_state(deform.optimizer.state_dict(),state['deform_optimizer'],'deform_optimizer')
        equal_state(deform.grid4d.state_dict(),state['hash'],'hash')
        equal_state(deform.deform.state_dict(),state['deform'],'deform')
        equal_state(rng_state(),state['rng'],'rng')
        for name in ['max_radii2D','xyz_gradient_accum','denom']:
            equal_state(getattr(gs,name),state['statistics'][name],name)
        self.install_events(gs)
        save(self.run/'resume.json',dict(at=stamp(),source=str(self.resume_path),after_iteration=state['iteration'],
             state='gaussians_hash_deform_optimizers_densification_rng_and_camera_stack_restored',
             exact_tensor_optimizer_rng_equality_verified=True))
        return state['iteration']+1,stack

    def install_events(self, gs):
        # Observe calls without changing the optimizer's clone/split/prune decisions.
        prune, postfix = gs.prune_points, gs.densification_postfix
        def record_prune(mask):
            count=int(mask.sum())
            evidence=None  # Retain structural counts; omit unused per-event point dumps.
            self.events.append(dict(iteration=self.current,event='prune',removed=count,points_before=len(gs.get_xyz),evidence=evidence))
            return prune(mask)
        def record_add(*args):
            evidence=None  # No additional diagnostic deformation/render before mutation.
            self.events.append(dict(iteration=self.current,event='append',added=len(args[0]),points_before=len(gs.get_xyz),evidence=evidence))
            return postfix(*args)
        gs.prune_points, gs.densification_postfix = record_prune, record_add

    @torch.no_grad()
    def record_structure_points(self, xyz, kind):
        cam=self.current_cam
        world=xyz
        if self.current>=self.opt.warm_up:
            flags=self.deform.reg_spatial_able,self.deform.reg_temporal_able
            self.deform.reg_spatial_able=self.deform.reg_temporal_able=False
            try:
                warp=self.deform.step(xyz,cam.fid.unsqueeze(0).expand(len(xyz),-1))['d_xyz']
                world=(native.quaternion_to_matrix(warp[:,:4])@xyz[...,None]).squeeze(-1)+warp[:,4:]
            finally:
                self.deform.reg_spatial_able,self.deform.reg_temporal_able=flags
        xyz1=torch.cat([world,torch.ones_like(world[:,:1])],1)
        clip=xyz1@cam.full_proj_transform
        uv=(clip[:,:2]/clip[:,3:]+1)*torch.tensor([cam.image_width,cam.image_height],device='cuda')/2-.5
        out=self.run/'diagnostics';out.mkdir(exist_ok=True)
        p=out/f'structure_{self.current:05d}_{len(self.events)}_{kind}.npz'
        np.savez_compressed(p,canonical_xyz=xyz.cpu().numpy(),world_xyz=world.cpu().numpy(),uv=uv.cpu().numpy(),
                            view=np.asarray(cam.source_view),frame=np.asarray(cam.source_frame))
        return str(p.relative_to(self.run))

    def before(self, iteration, stack, gs, deform):
        self.current = iteration
        self.iter_wall = time.monotonic()
        self.exclude_wall = 0.
        self.gpu_start.record()
        assert shutil.disk_usage(self.run).free > 8*2**30

    def backward(self, iteration, cam, gs, deform, package, loss):
        assert bool(torch.isfinite(loss)), f'Nonfinite loss at {iteration}'
        assert cam.source_view in self.config['invariants']['train_views']
        self.current_cam=cam
        assert bool(torch.isfinite(gs._xyz).all())
        self.counts['training_renders'] += 1
        self.counts['image_pixels_exposed'] += cam.image_width*cam.image_height
        self.counts['reference_foreground_pixels_exposed'] += cam.foreground_pixels
        key=f'{cam.source_view}:{cam.source_frame}'
        self.exposure[key]=self.exposure.get(key,0)+1
        if iteration in [1,self.opt.warm_up,self.opt.warm_up+1] or iteration % 100 == 0:
            groups={}
            for name,params in [('gaussian',[getattr(gs,n) for n in PARAMETERS]),('hash',deform.grid4d.parameters()),('deform',deform.deform.parameters())]:
                grads=[p.grad for p in params if p.grad is not None]
                assert all(bool(torch.isfinite(g).all()) for g in grads), (iteration,name)
                groups[name]=sum(float(g.abs().sum()) for g in grads)
            save(self.run/'gradient_latest.json',dict(iteration=iteration,abs_gradient_sum=groups))

    def saved(self, iteration, scene, gs, deform):
        started=time.monotonic()
        saved_rng=rng_state()
        flags=deform.reg_spatial_able,deform.reg_temporal_able
        deform.reg_spatial_able=deform.reg_temporal_able=False
        refs={}
        try:
            with torch.no_grad():
                for view in [1,9,17]:
                    for frame in [0,(scene.data['frame_count']-1)/2,scene.data['frame_count']-1,12.25]:
                        cam=camera(scene.data,view,frame)
                        rgb,alpha=raw_render(gs,deform,SimpleNamespace(**self.config['pipeline']),cam)
                        assert bool(torch.isfinite(rgb).all()) and bool(torch.isfinite(alpha).all())
                        refs[f'v{view}_f{frame:g}']=torch.cat([rgb,alpha]).cpu().numpy()
            root=self.run/'snapshots';root.mkdir(exist_ok=True)
            np.savez_compressed(root/f'{iteration}_reference.npz',**refs)
            files=[self.run/'point_cloud'/f'iteration_{iteration}'/'point_cloud.ply',self.run/'deform'/f'iteration_{iteration}'/'deform.pth']
            save(root/f'{iteration}.json',dict(iteration=iteration,actual_optimizer_updates=iteration-1,
                active_sh_degree=gs.active_sh_degree,points=len(gs.get_xyz),files={str(p.relative_to(self.run)):sha(p) for p in files},
                counts_before_current_step=dict(self.counts),reference_views=[1,9,17],
                reference_scope='training cameras only; independent replay at endpoints,midpoint,non-slot12.25',
                state='native_before_optimizer_snapshot',time_mapping='t=f/(F-1)'))
        finally:
            deform.reg_spatial_able,deform.reg_temporal_able=flags
            set_rng(saved_rng)
        self.exclude_wall += time.monotonic()-started

    def checkpoint(self, iteration, gs, deform, stack):
        atomic_torch(self.run/'recovery.pt',dict(iteration=iteration,config_sha256=sha(self.run/'config.json'),
            gaussians={n:getattr(gs,n).detach() for n in PARAMETERS},gaussian_optimizer=gs.optimizer.state_dict(),
            statistics={n:getattr(gs,n) for n in ['max_radii2D','xyz_gradient_accum','denom']},
            active_sh_degree=gs.active_sh_degree,hash=deform.grid4d.state_dict(),deform=deform.deform.state_dict(),
            deform_optimizer=deform.optimizer.state_dict(),rng=rng_state(),stack=None if stack is None else [int(c.uid) for c in stack],
            counts=self.counts,exposure=self.exposure,wall_training=self.wall_training,gpu_timeline_ms=self.gpu_timeline_ms))

    def after(self, iteration, cam, gs, deform, loss, updated, stack):
        self.gpu_end.record();self.gpu_end.synchronize()
        self.gpu_timeline_ms+=self.gpu_start.elapsed_time(self.gpu_end)
        self.wall_training+=time.monotonic()-self.iter_wall-self.exclude_wall
        if updated:
            self.counts['optimizer_calls']+=1
            self.counts['dynamic_gradient_updates']+=int(iteration>=self.opt.warm_up)
        record=dict(iteration=iteration,loss=float(loss),view=cam.source_view,frame=cam.source_frame,
            points=len(gs.get_xyz),updated=updated,wall_training_seconds=self.wall_training,
            current_lr={g['name']:g['lr'] for g in gs.optimizer.param_groups+deform.optimizer.param_groups})
        self.trace.write(json.dumps(record,allow_nan=False)+'\n')
        if self.events:
            with (self.run/'structure_events.jsonl').open('a') as f:
                f.writelines(json.dumps(e)+'\n' for e in self.events)
            self.events.clear()
        if iteration%100==0 or iteration in [1,self.opt.warm_up,self.stop_after,self.opt.iterations]:
            save(self.run/'progress.json',dict(state='training',updated_at=stamp(),pid=os.getpid(),device=self.device,
                **record,counts=self.counts,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                gpu_timeline_ms=self.gpu_timeline_ms,gpu_timing_scope='CUDA timeline includes CPU submission gaps and snapshot I/O; not pure GPU busy time'))
        if iteration%1000==0 or iteration in [self.stop_after,self.opt.iterations] or (self.config['smoke'] and iteration==60):
            self.checkpoint(iteration,gs,deform,stack)
            if self.config['smoke'] and iteration==60:
                shutil.copyfile(self.run/'recovery.pt',self.run/'recovery60.pt')
            save(self.run/'sample_exposure.json',self.exposure)
        if iteration==self.stop_after:
            raise StopRequested()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--run',required=True)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--resume-from');parser.add_argument('--stop-after',type=int)
    args=parser.parse_args();device=verify()
    run=Path(args.run).resolve();assert run.is_relative_to(ROOT)
    config=json.loads(Path(args.config).read_text())
    assert not (args.resume and args.resume_from)
    if args.resume:
        existing=json.loads((run/'config.json').read_text());assert existing==config
        assert not (run/'complete.json').exists()
    else:
        run.mkdir(parents=True,exist_ok=False);save(run/'config.json',config)
    sys.addaudithook(protect_inputs)
    seed=config['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.cuda.reset_peak_memory_stats()
    dataset=SimpleNamespace(**dict(config['dataset'],model_path=str(run)))
    camera_source=json.loads(Path(config['provenance']['manifest']).read_text())
    save(run/'asset_camera_data.json',{k:camera_source[k] for k in ['cameras','width','height','frame_count',
         'time_denominator','train_views','heldout_views','source_to_normalized']})
    resume_path=run/'recovery.pt' if args.resume else Path(args.resume_from).resolve() if args.resume_from else None
    if resume_path:
        assert resume_path.is_relative_to(ROOT) and resume_path.is_file()
    monitor=Monitor(config,run,device,resume_path,args.stop_after)
    native.Scene=KnownScene
    native.prepare_output_and_logger=lambda d:(None,d)
    native.training_report=lambda *args,**kwargs:None
    native._study_resume=monitor.initialize
    native._study_before_iteration=monitor.before
    native._study_after_backward=monitor.backward
    native._study_saved=monitor.saved
    native._study_after_iteration=monitor.after
    save(run/'start.json',dict(at=stamp(),pid=os.getpid(),device=device,smoke=config['smoke'],resume=args.resume,
        command=sys.argv,config_sha256=sha(run/'config.json'),source_patch_sha256=sha(ROOT/'research/source.patch')))
    try:
        native.training(dataset,SimpleNamespace(**config['optimization']),SimpleNamespace(**config['pipeline']),[],config['saving_iterations'])
    except StopRequested:
        save(run/'interrupted.json',dict(state='explicit_recovery_smoke_stop',iteration=monitor.current,at=stamp()))
        return
    finally:
        monitor.trace.close()
    assert monitor.counts['optimizer_calls']==config['optimization']['iterations']-1
    assert monitor.counts['dynamic_gradient_updates']==config['invariants']['dynamic_actual_updates']
    save(run/'complete.json',dict(state='optimization_complete_replay_and_quality_pending',at=stamp(),counts=monitor.counts,
        training_wall_seconds=monitor.wall_training,device=device,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        smoke=config['smoke'],snapshots=config['saving_iterations']))


if __name__=='__main__':
    main()
