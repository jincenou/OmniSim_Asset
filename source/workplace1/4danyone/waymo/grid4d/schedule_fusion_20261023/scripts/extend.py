"""Pre-registered conditional5k continuation of a single jointly selected recipe."""
import argparse,importlib.util,json,os,random,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
STUDY=Path(__file__).resolve().parents[1];GRID=STUDY.parent
sys.path.insert(0,str(GRID/'scripts'))
from grid_common import PARAMETERS,new_deform,GaussianModel,rng_state,set_rng,sha,save,render,raw_render,camera,verify
from optimize_grid import equal
from train_grid import atomic_torch,protect_inputs
sys.path.insert(0,str(GRID/'optimization_cross8_20261018/scripts'))
from quality_common import restore,load_train
from utils.loss_utils import ssim


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--label',required=True);ap.add_argument('--arm',required=True,choices=['F1','F2']);args=ap.parse_args()
    begin=time.monotonic();device=verify();sys.addaudithook(protect_inputs)
    decision=json.loads((STUDY/'DECISION_33K.json').read_text());assert decision['selected_for_extension']==args.arm
    source=STUDY/'runs'/f'{args.label}_{args.arm}';run=STUDY/'runs'/f'{args.label}_{args.arm}38';run.mkdir(parents=True,exist_ok=False)
    model,deform,pipe,config,data,state=restore(source)
    assert state['actual_updates']==32999 and state['plan_sha256']==sha(STUDY/'PLAN.json')
    save(run/'config.json',config);save(run/'asset_camera_data.json',data)
    cams,alphas,paths=load_train(data,args.label);by_uid={int(c.uid):i for i,c in enumerate(cams)};stack=list(state['stack'])
    constants={g['name']:g['lr']/.25 for o in [model.optimizer,deform.optimizer] for g in o.param_groups}
    set_rng(state['rng']);save(run/'restore.json',{'state':'full_state_exact','parent_sha256':sha(source/'recovery.pt'),'actual_updates':32999,'plan_sha256':sha(STUDY/'PLAN.json')})
    opt=SimpleNamespace(**config['optimization']);background=torch.full((3,),.5,device='cuda');sequence=[];N=len(model.get_xyz)
    load_seconds=time.monotonic()-begin;torch.cuda.synchronize();loop=time.monotonic()
    for step in range(1,5001):
        if not stack:stack=list(by_uid)
        uid=stack.pop(random.randrange(len(stack)));i=by_uid[uid];cam=cams[i];sequence.append(uid)
        model.update_learning_rate(33000+step);deform.update_learning_rate(33000+step)
        for o in [model.optimizer,deform.optimizer]:
            for g in o.param_groups:g['lr']=(g['lr'] if g['name'] in ['xyz','deform','grid'] else constants[g['name']])*.25
        warp=deform.step(model.get_xyz.detach(),cam.fid.unsqueeze(0).expand(N,-1))
        rgb=render(cam,model,pipe,background,warp['d_xyz'],warp['d_rotation'],warp['d_scaling'])['render']
        alpha=render(cam,model,pipe,torch.zeros_like(background),warp['d_xyz'],warp['d_rotation'],warp['d_scaling'],override_color=torch.ones_like(model.get_xyz))['render'][0]
        target=alphas[i].float()/255;fg=target>.5;error=(alpha-target).abs();alpha_loss=.5*(error[fg].mean()+error[~fg].mean())
        reg=warp['reg_spatial'].mean()*opt.lambda_spatial_tv+warp['reg_temporal'].mean()*opt.lambda_temporal_tv
        loss=.8*(rgb-cam.original_image).abs().mean()+.2*(1-ssim(rgb,cam.original_image))+.1*alpha_loss+reg
        assert bool(torch.isfinite(loss));loss.backward();model.optimizer.step();deform.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True);deform.optimizer.zero_grad(set_to_none=True)
        assert len(model.get_xyz)==N
        if step%500==0:save(run/'progress.json',{'state':'training','iteration':step,'actual_updates':32999+step,'loss':float(loss),'pid':os.getpid(),'device':device})
    torch.cuda.synchronize();train_seconds=time.monotonic()-loop;save_start=time.monotonic();saved=rng_state()
    model.save_ply(str(run/'point_cloud/iteration_38000/point_cloud.ply'));deform.save_weights(str(run),38000)
    flags=deform.reg_spatial_able,deform.reg_temporal_able;deform.reg_spatial_able=deform.reg_temporal_able=False
    refs={}
    try:
        with torch.no_grad():
            for v in [1,9,17]:
                for f in [0,(data['frame_count']-1)/2,data['frame_count']-1,12.25]:
                    rgb,alpha=raw_render(model,deform,pipe,camera(data,v,f));refs[f'v{v}_f{f:g}']=torch.cat([rgb,alpha]).cpu().numpy()
    finally:deform.reg_spatial_able,deform.reg_temporal_able=flags
    (run/'snapshots').mkdir();np.savez_compressed(run/'snapshots/38000_reference.npz',**refs)
    save(run/'snapshots/38000.json',{'iteration':38000,'actual_optimizer_updates':37999,'active_sh_degree':model.active_sh_degree,'points':N,'state':'after_actual_optimizer_update'})
    counts=dict(state['counts'])
    for key,increment in [('optimizer_calls',5000),('dynamic_gradient_updates',5000),('training_renders',10000),('backward_calls',5000)]:counts[key]+=increment
    set_rng(saved);state.update(iteration=38000,actual_updates=37999,arm=args.arm+'38',counts=counts,
        gaussians={n:getattr(model,n).detach() for n in PARAMETERS},gaussian_optimizer=model.optimizer.state_dict(),
        statistics={n:getattr(model,n) for n in ['max_radii2D','xyz_gradient_accum','denom']},
        hash=deform.grid4d.state_dict(),deform=deform.deform.state_dict(),deform_optimizer=deform.optimizer.state_dict(),rng=saved,stack=stack)
    atomic_torch(run/'recovery.pt',state);equal(rng_state(),saved)
    save(run/'sampling.json',sequence)
    save(run/'complete.json',{'state':'training_complete_replay_pending','label':args.label,'arm':args.arm+'38','actual_updates':37999,
        'new_updates':5000,'points':N,'load_seconds':load_seconds,'train_seconds':train_seconds,'save_seconds':time.monotonic()-save_start,
        'process_seconds':time.monotonic()-begin,'device':device,'parent':str(source),'plan_sha256':sha(STUDY/'PLAN.json'),
        'new_counts':{'optimizer_calls':5000,'backward_calls':5000,'training_renders':10000}})
    print(args.label,args.arm+'38','complete',flush=True)


if __name__=='__main__':main()
