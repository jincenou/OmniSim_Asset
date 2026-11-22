"""Exact alpha15k and A5k objectives with generic source/output, paired frozen sampling."""
import argparse,os,shutil
from quality_common import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);parser.add_argument('--stage',choices=['alpha','A'],required=True);args=parser.parse_args()
    begin=time.monotonic();device=verify();spec=plan();sys.addaudithook(protect_inputs)
    source=NEW/'runs'/f'{args.label}_{"33k" if args.stage=="alpha" else "alpha"}'
    run=NEW/'runs'/f'{args.label}_{args.stage}';assert (source/'complete.json').exists();run.mkdir(exist_ok=False)
    model,deform,pipe,config,data,state=restore(source);base=32999 if args.stage=='alpha' else 47999;budget=15000 if args.stage=='alpha' else 5000
    actual_start=state['counts']['optimizer_calls'] if args.stage=='alpha' else state['global_actual_updates'];assert actual_start==base
    original_group_steps={g['name']:[float(o.state[p]['step']) for p in g['params']] for o in [model.optimizer,deform.optimizer] for g in o.param_groups}
    config.update(stage=args.stage,parent=str(source),quality_plan_sha256=sha(NEW/'PLAN.json'));save(run/'config.json',config);save(run/'asset_camera_data.json',data)
    cams,alphas,paths=load_train(data,args.label);save(run/'training_image_paths.json',paths);by_uid={int(c.uid):i for i,c in enumerate(cams)};all_uids=list(by_uid)
    stack=[] if state['stack'] is None else list(state['stack']);sampler=random.Random();sampler.setstate(state['rng']['python']);probe_stack=list(stack);sequence=[]
    for step in range(budget):
        if not probe_stack:probe_stack=list(all_uids)
        sequence.append(probe_stack.pop(sampler.randrange(len(probe_stack))))
    save(run/'sampling.json',sequence)
    base_lrs={g['name']:g['lr'] for o in [model.optimizer,deform.optimizer] for g in o.param_groups};lrs=[]
    for j in range(budget):
        mapped=j if args.stage=='alpha' else int(14999*j/4999);virtual=15000+mapped*17999/14999
        model.update_learning_rate(virtual);deform.update_learning_rate(virtual)
        row={g['name']:g['lr'] if g['name'] in ['xyz','deform','grid'] else base_lrs[g['name']] for o in [model.optimizer,deform.optimizer] for g in o.param_groups}
        lrs.append({k:v*(.25 if args.stage=='A' else 1) for k,v in row.items()})
    if args.stage=='A':
        old=json.loads((PREVIOUS/'data/N04/learning_rates.json').read_text());assert all(all(r[k]==old[j][k] for k in r) for j,r in enumerate(lrs))
    save(run/'learning_rates.json',lrs)
    set_rng(state['rng']);save(run/'restore_validation.json',dict(state='passed',full_state_exact=True,source_recovery_sha256=sha(source/'recovery.pt'),global_actual_updates=actual_start,device=device));del state
    opt=SimpleNamespace(**config['optimization']);n=len(model.get_xyz);background=torch.full((3,),.5,device='cuda');torch.cuda.reset_peak_memory_stats()
    snapshots=spec['cross_person']['save_alpha_extra' if args.stage=='alpha' else 'save_A_extra'];counts=dict(training_loop_seconds=0.,snapshot_seconds=0.,render_calls=0,backward_calls=0,updates=0)
    loop=time.monotonic()
    with (run/'trace.jsonl').open('x',buffering=1) as trace:
        for step in range(1,budget+1):
            tick=time.monotonic()
            if not stack:stack=list(all_uids)
            uid=stack.pop(random.randrange(len(stack)));assert uid==sequence[step-1];i=by_uid[uid];cam=cams[i]
            for optimizer in [model.optimizer,deform.optimizer]:
                for group in optimizer.param_groups:group['lr']=lrs[step-1][group['name']]
            warp=deform.step(model.get_xyz.detach(),cam.fid.unsqueeze(0).expand(n,-1))
            rgb=render(cam,model,pipe,background,warp['d_xyz'],warp['d_rotation'],warp['d_scaling'])['render']
            alpha=render(cam,model,pipe,torch.zeros_like(background),warp['d_xyz'],warp['d_rotation'],warp['d_scaling'],override_color=torch.ones_like(model.get_xyz))['render'][0]
            delta=(rgb-cam.original_image).abs();l1=delta.mean(0).mean() if args.stage=='alpha' else delta.mean()
            rgb_loss=.8*l1+.2*(1-ssim(rgb,cam.original_image));alpha_loss=old_alpha(alpha,alphas[i].float()/255)
            reg=warp['reg_spatial'].mean()*opt.lambda_spatial_tv+warp['reg_temporal'].mean()*opt.lambda_temporal_tv
            loss=rgb_loss+.1*alpha_loss+reg;assert bool(torch.isfinite(loss));loss.backward()
            model.optimizer.step();deform.optimizer.step();model.optimizer.zero_grad(set_to_none=True);deform.optimizer.zero_grad(set_to_none=True)
            assert len(model.get_xyz)==n;torch.cuda.synchronize();counts['training_loop_seconds']+=time.monotonic()-tick;counts['render_calls']+=2;counts['backward_calls']+=1;counts['updates']+=1
            row=dict(step=step,actual_updates=base+step,uid=uid,view=cam.source_view,frame=cam.source_frame,loss=float(loss),rgb_loss=float(rgb_loss),alpha_loss=float(alpha_loss),regularization=float(reg),points=n,learning_rates=lrs[step-1],wall_seconds=time.monotonic()-loop)
            trace.write(json.dumps(row,allow_nan=False)+'\n')
            if step%100==0 or step==1:
                assert shutil.disk_usage(NEW).free>8*2**30
                save(run/'progress.json',dict(state='training',pid=os.getpid(),device=device,costs=counts,**row))
            if step in snapshots:
                for o in [model.optimizer,deform.optimizer]:
                    for g in o.param_groups:assert [float(o.state[p]['step']) for p in g['params']]==[s+step for s in original_group_steps[g['name']]]
                nominal=base+step+1;counts['snapshot_seconds']+=capture(run,nominal,base+step,step,model,deform,pipe,data,stack,counts,source)
                save(run/'budget'/f'{nominal}.json',dict(actual_updates=base+step,stage_updates=step,loop_costs=counts,end_to_end_seconds_so_far=time.monotonic()-begin))
    assert random.getstate()==sampler.getstate() and stack==probe_stack
    save(run/'complete.json',dict(state='training_complete_replay_pending',label=args.label,stage=args.stage,total_actual_updates=base+budget,extra_updates=budget,points=n,device=device,costs=counts,end_to_end_seconds=time.monotonic()-begin,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),sampling_exact=True,sampling_sha256=sha(run/'sampling.json'),lr_sha256=sha(run/'learning_rates.json')))
    print(args.label,args.stage,'completed',base+budget,flush=True)

if __name__=='__main__':main()
