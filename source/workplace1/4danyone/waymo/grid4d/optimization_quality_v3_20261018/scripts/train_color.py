"""Paired SH-only Adam and fixed-block residual updates, bounded by frozen PLAN."""
import argparse,math
from color_common import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['C1','C2'],required=True);args=ap.parse_args();begin=time.monotonic();device=verify();sys.addaudithook(protect_inputs)
    assert json.loads((NEW/'color_preflight.json').read_text())['state']=='passed'
    source,model,deform,pipe,config,data,frozen=setup();observations=pool(data);run=NEW/'runs'/f'N11_{args.arm}';run.mkdir(exist_ok=False)
    config.update(color_arm=args.arm,color_base_updates=52999,color_plan_sha256=sha(NEW/'PLAN.json'));save(run/'config.json',config);save(run/'asset_camera_data.json',data);save(run/'training_image_paths.json',[p for c,p in observations]);save(run/'color_pool.json',[dict(view=c.source_view,frame=c.source_frame,path=p,sha256=sha(p)) for c,p in observations])
    lrs=json.loads((PREVIOUS/'data/N11/learning_rates.json').read_text())[0]
    opt=torch.optim.Adam([{'params':[model._features_dc],'lr':lrs['f_dc']},{'params':[model._features_rest],'lr':lrs['f_rest']}],eps=1e-15) if args.arm=='C1' else None
    costs=dict(training_render_calls=0,training_backward_calls=0,color_updates=0,statistics_seconds=0.,solve_seconds=0.,full_pool_evaluation_seconds=0.,snapshot_seconds=0.,loop_seconds=0.)
    train_start=time.monotonic();torch.cuda.reset_peak_memory_stats();history=[];epoch=0;truncated=False
    costs['snapshot_seconds']+=save_color(run,0,0,model,deform,pipe,data,opt,costs)
    tick=time.monotonic();initial=full_loss(model,deform,pipe,observations);costs['full_pool_evaluation_seconds']+=time.monotonic()-tick;history.append(dict(color_pass=0,mse=initial))
    if args.arm=='C2':
        tick=time.monotonic();n=len(model.get_xyz);k=16;blocks=torch.zeros(n,k,k,device='cuda',dtype=torch.float64)
        for cam,path in observations:
            v,y=contribution(model,deform,pipe,cam);blocks.add_(v.double()[:,None,None]*(y.double()[:,:,None]*y.double()[:,None,:]));costs['training_render_calls']+=1;costs['training_backward_calls']+=1
        blocks.mul_(2/(3*data['height']*data['width']*len(observations)));support=blocks.diagonal(dim1=-2,dim2=-1).sum(-1)>0
        damping=blocks.diagonal(dim1=-2,dim2=-1).sum(-1)/k*1e-4
        blocks.diagonal(dim1=-2,dim2=-1).add_(damping[:,None]);chol,info=torch.linalg.cholesky_ex(blocks);valid=support&(info==0);assert bool(valid.any())
        save(run/'block_statistics.json',dict(first_order_contribution=True,exact_hessian=False,normalization=2/(3*data['height']*data['width']*len(observations)),trace_relative_damping=1e-4,valid_blocks=int(valid.sum()),zero_support=int((~support).sum()),unsolved_blocks=int((support&(info!=0)).sum()),step=.5))
        costs['statistics_seconds']=time.monotonic()-tick;del blocks;epoch=1
        costs['snapshot_seconds']+=save_color(run,1,0,model,deform,pipe,data,opt,costs);history.append(dict(color_pass=1,mse=initial))
    rng=random.Random(42)
    while epoch<20:
        if time.monotonic()-train_start>=1800:truncated=True;break
        epoch+=1;tick=time.monotonic()
        if args.arm=='C1':
            order=list(range(len(observations)));rng.shuffle(order)
            for index in order:
                if time.monotonic()-train_start>=1800:truncated=True;break
                cam,path=observations[index];opt.zero_grad(set_to_none=True);rgb=package(model,deform,pipe,cam)['render'];loss=(rgb-cam.original_image).square().mean();assert bool(torch.isfinite(loss));loss.backward();opt.step();costs['color_updates']+=1;costs['training_render_calls']+=1;costs['training_backward_calls']+=1
        else:
            model._features_dc.grad=None;model._features_rest.grad=None
            for cam,path in observations:
                if time.monotonic()-train_start>=1800:truncated=True;break
                rgb=package(model,deform,pipe,cam)['render'];loss=(rgb-cam.original_image).square().mean()/len(observations);assert bool(torch.isfinite(loss));loss.backward();costs['training_render_calls']+=1;costs['training_backward_calls']+=1
            if not truncated:
                solve=time.monotonic();gradient=torch.cat([model._features_dc.grad,model._features_rest.grad],1).double();delta=torch.zeros_like(gradient);delta[valid]=torch.cholesky_solve(gradient[valid],chol[valid]);assert bool(torch.isfinite(delta).all())
                with torch.no_grad():model._features_dc.sub_(.5*delta[:,:1].float());model._features_rest.sub_(.5*delta[:,1:].float())
                costs['solve_seconds']+=time.monotonic()-solve;costs['color_updates']+=1
        torch.cuda.synchronize();costs['loop_seconds']+=time.monotonic()-tick;verify_frozen(model,deform,frozen)
        tick=time.monotonic();value=full_loss(model,deform,pipe,observations);costs['full_pool_evaluation_seconds']+=time.monotonic()-tick;history.append(dict(color_pass=epoch,mse=value,partial_pass=truncated,seconds=time.monotonic()-train_start));save(run/'mse_curve.json',history)
        save(run/'progress.json',dict(state='training' if not truncated else 'budget_truncated',device=device,pid=os.getpid(),color_pass=epoch,costs=costs,mse=value,elapsed_seconds=time.monotonic()-train_start));print(args.arm,epoch,value,flush=True)
        if epoch in [1,5,10,20] or truncated:costs['snapshot_seconds']+=save_color(run,epoch,costs['color_updates'],model,deform,pipe,data,opt,costs)
        if truncated:break
    if not (run/'snapshots'/f'{epoch}.json').exists():costs['snapshot_seconds']+=save_color(run,epoch,costs['color_updates'],model,deform,pipe,data,opt,costs)
    verify_frozen(model,deform,frozen)
    save(run/'complete.json',dict(state='budget_truncated' if truncated else 'color_training_complete_replay_quality_pending',arm=args.arm,label='N11',color_pass=epoch,base_grid4d_actual_updates=52999,costs=costs,device=device,frozen_parameters_exact=True,initial_full_pool_mse=initial,final_full_pool_mse=history[-1]['mse'],training_wall_seconds=time.monotonic()-train_start,end_to_end_seconds=time.monotonic()-begin,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),source=str(source)))
if __name__=='__main__':main()
