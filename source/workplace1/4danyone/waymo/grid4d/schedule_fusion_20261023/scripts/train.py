"""Frozen alpha/LR schedule comparison using the original native training loop."""
import argparse, hashlib, importlib.util, json, math, os, random, shutil, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image

STUDY=Path(__file__).resolve().parents[1]
BASE=STUDY.parent
sys.path.insert(0,str(BASE/'scripts'))
from grid_common import PARAMETERS, KnownScene, camera, raw_render, render, rng_state, set_rng, sha, save, verify, new_deform
from train_grid import protect_inputs, atomic_torch, StopRequested
from optimize_grid import equal


def schedule(plan, arm, iteration):
    if arm=='prefix':return 0.0,1.0
    if arm=='middle':arm='F0'
    s=plan['arms'][arm]
    alpha=.1*min(1.,max(0.,(iteration-s['alpha_start'])/(s['alpha_ramp_end']-s['alpha_start'])))
    if s['late_lr_start'] is None:return alpha,1.0
    fraction=min(1.,max(0.,(iteration-s['late_lr_start'])/(s['late_lr_end']-s['late_lr_start'])))
    return alpha,math.exp(math.log(s['late_lr_multiplier'])*fraction)


class AlphaScene(KnownScene):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        for cam,path in zip(self.train_cameras,self.loaded_paths):
            with Image.open(path) as image:
                cam.reference_alpha=torch.as_tensor(np.asarray(image)[...,3].copy(),device='cuda')


class Monitor:
    def __init__(self,args,plan,config,run,device):
        self.args,self.plan,self.config,self.run,self.device=args,plan,config,run,device
        self.begin=time.monotonic();self.current=0;self.weight=0.;self.loss_alpha=0.;self.factor=1.
        self.counts={'optimizer_calls':0,'training_renders':0,'backward_calls':0,'dynamic_gradient_updates':0}
        self.sampling=[];self.lr_rows=[];self.structure=[];self.stage_start=0;self.native_loss=None
        self.train_seconds=0.;self.save_seconds=0.;self.load_seconds=0.

    def initialize(self,scene,model,deform):
        self.scene,self.model,self.deform=scene,model,deform
        start=1;stack=None
        if self.args.source:
            path=Path(self.args.source)/'recovery.pt';state=torch.load(path,map_location='cpu',weights_only=False)
            assert state['plan_sha256']==sha(STUDY/'PLAN.json')
            assert state['iteration']==(28000 if self.args.arm in ['F0','F1'] else 10000)
            for name in PARAMETERS:setattr(model,name,torch.nn.Parameter(state['gaussians'][name].cuda()))
            model.training_setup(SimpleNamespace(**self.config['optimization']))
            model.optimizer.load_state_dict(state['gaussian_optimizer'])
            for name in ['max_radii2D','xyz_gradient_accum','denom']:setattr(model,name,state['statistics'][name].cuda())
            model.active_sh_degree=state['active_sh_degree'];deform.grid4d.load_state_dict(state['hash'],strict=True)
            deform.deform.load_state_dict(state['deform'],strict=True);deform.optimizer.load_state_dict(state['deform_optimizer'])
            for name in PARAMETERS:equal(getattr(model,name),state['gaussians'][name])
            equal(model.optimizer.state_dict(),state['gaussian_optimizer']);equal(deform.optimizer.state_dict(),state['deform_optimizer'])
            equal(deform.grid4d.state_dict(),state['hash']);equal(deform.deform.state_dict(),state['deform'])
            cams={int(c.uid):c for c in scene.getTrainCameras()}
            stack=None if state['stack'] is None else [cams[uid] for uid in state['stack']]
            set_rng(state['rng']);equal(rng_state(),state['rng'])
            self.counts=dict(state['counts']);self.stage_start=self.counts['optimizer_calls'];start=state['iteration']+1
            save(self.run/'restore.json',{'state':'full_state_exact','parent':str(path),'parent_sha256':sha(path),
                'gaussians_optimizers_statistics_hash_deform_RNG_stack':True,'actual_updates':self.stage_start})
        self.constant_lrs={g['name']:g['lr'] for o in [model.optimizer,deform.optimizer] for g in o.param_groups}
        self.install_structure(model)
        self.load_seconds=time.monotonic()-self.begin
        torch.cuda.synchronize();self.loop_start=time.monotonic()
        return start,stack

    def install_structure(self,model):
        prune,postfix=model.prune_points,model.densification_postfix
        def p(mask):
            self.structure.append({'iteration':self.current,'event':'prune','removed':int(mask.sum())})
            return prune(mask)
        def q(*args):
            self.structure.append({'iteration':self.current,'event':'append','added':len(args[0])})
            return postfix(*args)
        model.prune_points=p;model.densification_postfix=q

    def before(self,it,stack,model,deform):
        self.current=it;self.tick=time.monotonic()
        self.weight,self.factor=schedule(self.plan,self.args.arm,it)
        for o in [model.optimizer,deform.optimizer]:
            for g in o.param_groups:
                baseline=g['lr'] if g['name'] in ['xyz','deform','grid'] else self.constant_lrs[g['name']]
                g['lr']=baseline*self.factor
        if it%500==0 or it in [1,10001,15001,18000,28000,33000]:
            self.lr_rows.append({'iteration':it,'alpha_weight':self.weight,'factor':self.factor,
                'lr':{g['name']:g['lr'] for o in [model.optimizer,deform.optimizer] for g in o.param_groups}})

    def alpha(self,it,cam,model,pipe,dx,dr,ds):
        self.loss_alpha=0.
        if self.weight==0:return 0.
        target=cam.reference_alpha.float()/255;fg=target>.5
        assert bool(fg.any()) and bool((~fg).any())
        alpha=render(cam,model,pipe,torch.zeros(3,device='cuda'),dx,dr,ds,override_color=torch.ones_like(model.get_xyz))['render'][0]
        error=(alpha-target).abs();loss=.5*(error[fg].mean()+error[~fg].mean())
        self.loss_alpha=float(loss.detach());return self.weight*loss

    def backward(self,it,cam,model,deform,package,loss):
        assert cam.source_view in self.config['invariants']['train_views']
        self.counts['training_renders']+=1+int(self.weight>0)
        self.counts['backward_calls']+=1
        self.sampling.append([it,int(cam.uid)])
        if it%500==0:
            assert bool(torch.isfinite(loss))
            for parameters in [[getattr(model,n) for n in PARAMETERS],deform.grid4d.parameters(),deform.deform.parameters()]:
                assert all(bool(torch.isfinite(p.grad).all()) for p in parameters if p.grad is not None)

    def checkpoint(self,it,stack,final=False):
        begin=time.monotonic();saved=rng_state();m,d=self.model,self.deform
        snapshot=10000 if not final else 33000
        if final:
            m.save_ply(str(self.run/f'point_cloud/iteration_{snapshot}/point_cloud.ply'))
            d.save_weights(str(self.run),snapshot)
            flags=d.reg_spatial_able,d.reg_temporal_able;d.reg_spatial_able=d.reg_temporal_able=False
            refs={}
            try:
                with torch.no_grad():
                    for v in [1,9,17]:
                        for frame in [0,(self.scene.data['frame_count']-1)/2,self.scene.data['frame_count']-1,12.25]:
                            rgb,alpha=raw_render(m,d,SimpleNamespace(**self.config['pipeline']),camera(self.scene.data,v,frame))
                            refs[f'v{v}_f{frame:g}']=torch.cat([rgb,alpha]).cpu().numpy()
            finally:d.reg_spatial_able,d.reg_temporal_able=flags
            (self.run/'snapshots').mkdir(exist_ok=True)
            np.savez_compressed(self.run/f'snapshots/{snapshot}_reference.npz',**refs)
            save(self.run/f'snapshots/{snapshot}.json',{'iteration':snapshot,'actual_optimizer_updates':self.counts['optimizer_calls'],
                'active_sh_degree':m.active_sh_degree,'points':len(m.get_xyz),'state':'after_actual_optimizer_update',
                'reference_views':[1,9,17],'reference_scope':'training views only, includes fractional12.25'})
        set_rng(saved)
        state={'iteration':it,'plan_sha256':sha(STUDY/'PLAN.json'),'gaussians':{n:getattr(m,n).detach() for n in PARAMETERS},
            'gaussian_optimizer':m.optimizer.state_dict(),'statistics':{n:getattr(m,n) for n in ['max_radii2D','xyz_gradient_accum','denom']},
            'active_sh_degree':m.active_sh_degree,'hash':d.grid4d.state_dict(),'deform':d.deform.state_dict(),
            'deform_optimizer':d.optimizer.state_dict(),'rng':saved,'stack':None if stack is None else [int(c.uid) for c in stack],
            'counts':dict(self.counts),'actual_updates':self.counts['optimizer_calls'],'label':self.args.label,'arm':self.args.arm}
        atomic_torch(self.run/'recovery.pt',state);equal(rng_state(),saved)
        self.save_seconds+=time.monotonic()-begin

    def after(self,it,cam,model,deform,loss,updated,stack):
        self.counts['optimizer_calls']+=int(updated)
        self.counts['dynamic_gradient_updates']+=int(updated and it>=self.config['optimization']['warm_up'])
        self.train_seconds+=time.monotonic()-self.tick
        stop=(self.args.arm=='prefix' and it==10000) or (self.args.arm=='middle' and it==28000)
        end=it==33000
        if it%500==0 or stop or end:
            assert shutil.disk_usage(STUDY).free>2**30,'Less than1GiB free'
            torch.cuda.synchronize()
            save(self.run/'progress.json',{'state':'training','iteration':it,'pid':os.getpid(),'device':self.device,
                'actual_updates':self.counts['optimizer_calls'],'alpha_weight':self.weight,'lr_factor':self.factor,
                'points':len(model.get_xyz),'loss':float(loss),'loop_wall_seconds':time.monotonic()-self.loop_start,
                'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()})
        if stop or end:
            self.checkpoint(it,stack,final=end)
            save(self.run/'sampling.json',self.sampling);save(self.run/'learning_rates.json',self.lr_rows)
            save(self.run/'structure.json',self.structure)
            save(self.run/'complete.json',{'state':'shared_state_complete' if stop else 'training_complete_replay_pending',
                'label':self.args.label,'arm':self.args.arm,'actual_updates':self.counts['optimizer_calls'],
                'new_updates':self.counts['optimizer_calls']-self.stage_start,'points':len(model.get_xyz),'counts':self.counts,
                'load_seconds':self.load_seconds,'train_seconds':self.train_seconds,'save_seconds':self.save_seconds,
                'process_seconds':time.monotonic()-self.begin,'device':self.device,'plan_sha256':sha(STUDY/'PLAN.json'),
                'sampling_sha256':sha(self.run/'sampling.json'),'peak_allocated':torch.cuda.max_memory_allocated()})
            if stop:raise StopRequested()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--label',required=True);ap.add_argument('--arm',required=True)
    ap.add_argument('--source');args=ap.parse_args();begin=time.monotonic()
    plan=json.loads((STUDY/'PLAN.json').read_text());assert sha(STUDY/'PLAN.json')==(STUDY/'PLAN.sha256').read_text().strip()
    assert sha(STUDY/'source/train_fused.py')==plan['code_sha256']
    assert args.label in plan['subjects'] and args.arm in ['prefix','middle',*plan['arms']]
    device=verify();config=json.loads((STUDY/'configs'/f'{args.label}.json').read_text())
    run=STUDY/'runs'/f'{args.label}_{args.arm}';run.mkdir(parents=True,exist_ok=False)
    save(run/'config.json',config);sys.addaudithook(protect_inputs)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False;torch.cuda.reset_peak_memory_stats()
    full=json.loads(Path(config['provenance']['manifest']).read_text())
    assert sha(config['provenance']['manifest'])==config['provenance']['manifest_sha256']
    save(run/'asset_camera_data.json',{k:full[k] for k in ['cameras','width','height','frame_count','time_denominator','train_views','heldout_views','source_to_normalized']})
    spec=importlib.util.spec_from_file_location('frozen_native_schedule',STUDY/'source/train_fused.py')
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    native.Scene=AlphaScene;native.prepare_output_and_logger=lambda dataset:(None,dataset)
    native.training_report=lambda *a,**kw:None
    from tqdm import tqdm
    native.tqdm=lambda *a,**kw:tqdm(*a,**dict(kw,disable=True))
    monitor=Monitor(args,plan,config,run,device)
    native._study_resume=monitor.initialize;native._study_before_iteration=monitor.before
    native._study_after_backward=monitor.backward;native._study_after_iteration=monitor.after
    native._fusion_alpha=monitor.alpha
    try:
        native.training(SimpleNamespace(**dict(config['dataset'],model_path=str(run))),SimpleNamespace(**config['optimization']),SimpleNamespace(**config['pipeline']),[],[])
    except StopRequested:
        assert args.arm in ['prefix','middle'] and monitor.counts['optimizer_calls']==(10000 if args.arm=='prefix' else 28000)
    else:assert monitor.counts['optimizer_calls']==32999
    save(run/'process_complete.json',{'seconds':time.monotonic()-begin,'state':'passed'})


if __name__=='__main__':main()
