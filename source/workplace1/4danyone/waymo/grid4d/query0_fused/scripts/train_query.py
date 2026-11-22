"""Fixed-capacity 2000-update continuation; only native observation selection changes."""
import argparse,hashlib,os,shutil
from quality_common import *
from selector import Selector
from demand import score

def snapshot(run,base,step,model,deform,pipe,data,stack,counts,parent,selector,base_sampler,table_path):
    begin=time.monotonic();iteration=base+step;saved=rng_state();ply=run/'point_cloud'/f'iteration_{iteration}'/'point_cloud.ply';model.save_ply(str(ply));deform.save_weights(str(run),iteration)
    refs=asset_reference(model,deform,pipe,data);np.savez_compressed(run/'snapshots'/f'{iteration}_reference.npz',**refs)
    payload=dict(global_actual_updates=iteration,stage_update=step,gaussians={n:getattr(model,n).detach() for n in PARAMETERS},gaussian_optimizer=model.optimizer.state_dict(),statistics={n:getattr(model,n) for n in ['max_radii2D','xyz_gradient_accum','denom']},active_sh_degree=model.active_sh_degree,hash=deform.grid4d.state_dict(),deform=deform.deform.state_dict(),deform_optimizer=deform.optimizer.state_dict(),rng=saved,stack=list(stack),costs=dict(counts),parent=str(parent),query_selector=selector.state(),base_sampler_state=base_sampler.getstate(),demand_table=str(table_path),demand_table_sha256=sha(table_path),plan_sha256=sha(NEW/'PLAN.json'))
    path=run/'recoveries'/f'{iteration}.pt';atomic_torch(path,payload);link=run/'recovery.next';link.symlink_to(path.relative_to(run));link.replace(run/'recovery.pt')
    files=[ply,run/'deform'/f'iteration_{iteration}'/'deform.pth',path]
    save(run/'snapshots'/f'{iteration}.json',dict(iteration=iteration,actual_optimizer_updates=iteration,stage_update=step,active_sh_degree=model.active_sh_degree,points=len(model.get_xyz),files={str(p.relative_to(run)):sha(p) for p in files},state='after_exact_actual_update',reference_views=[1,9,17]))
    equal(rng_state(),saved);return time.monotonic()-begin

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);parser.add_argument('--arm',choices=['U','P','E','Q'],required=True);parser.add_argument('--anchor',choices=['fused','A'],default='fused');args=parser.parse_args();begin=time.monotonic();device=verify();spec=plan();sys.addaudithook(protect_inputs)
    implementation=json.loads((NEW/'IMPLEMENTATION.json').read_text())
    for name,h in implementation['files'].items():assert sha(NEW/name)==h,name
    item=spec['anchors'][args.label][args.anchor];source=Path(item['run']);assert sha(source/'recovery.pt')==item['recovery_sha256']
    initial=NEW/'demand_tables'/f'{args.label}_{args.anchor}_initial.json';table=json.loads(initial.read_text());assert table['anchor_snapshot_sha256']==item['snapshot_sha256']
    review=json.loads((NEW/'research/candidate_visual_review.json').read_text());assert args.label in review['reviewed_labels']
    name=f'{args.label}_{args.anchor}_{args.arm}';run=NEW/'runs'/name;run.mkdir(exist_ok=False)
    for folder in ['snapshots','recoveries','budget']:(run/folder).mkdir()
    model,deform,pipe,config,data,state=restore(source);base=32999 if args.anchor=='fused' else 52999;assert state.get('global_actual_updates',state.get('counts',{}).get('optimizer_calls'))==base
    original_steps={g['name']:[float(o.state[p]['step']) for p in g['params']] for o in [model.optimizer,deform.optimizer] for g in o.param_groups};n=len(model.get_xyz);assert n==item['points'];degree=model.active_sh_degree
    config.update(stage='query0',anchor=args.anchor,arm=args.arm,parent=str(source),query_plan_sha256=sha(NEW/'PLAN.json'),actual_update_directory_names=True);save(run/'config.json',config);save(run/'asset_camera_data.json',data)
    cams,alphas,paths=load_train(data,args.label);save(run/'training_image_paths.json',paths);by_uid={int(c.uid):i for i,c in enumerate(cams)};all_uids=list(by_uid)
    stack=list(state['stack']) if state['stack'] is not None else [];base_sampler=random.Random();base_sampler.setstate(state['rng']['python']);generator=random.Random();generator.setstate(base_sampler.getstate());temp_stack=list(stack);base_sequence=[]
    for _ in range(2000):
        if not temp_stack:temp_stack=list(all_uids)
        base_sequence.append(temp_stack.pop(generator.randrange(len(temp_stack))))
    save(run/'base_sequence.json',base_sequence);candidate=json.loads((NEW/'data'/args.label/'candidates.json').read_text());selector=Selector(candidate,base_sequence,args.arm);selector.update(table)
    lrs=json.loads((NEW/'data/learning_rates.json').read_text());assert len(lrs)==2000;assert sha(NEW/'data/learning_rates.json')==spec['optimization']['schedule_sha256']
    save(run/'learning_rates.json',lrs);set_rng(state['rng']);save(run/'restore_validation.json',dict(state='passed',full_state_exact=True,parent_recovery_sha256=item['recovery_sha256'],base_actual_updates=base,device=device,initial_opacity_active_gt0005=int((model.get_opacity>.005).sum()),points=n));del state
    opt=SimpleNamespace(**config['optimization']);background=torch.full((3,),.5,device='cuda');torch.cuda.reset_peak_memory_stats();loop=time.monotonic();counts=dict(training_loop_seconds=0.,snapshot_seconds=0.,diagnostic_seconds=0.,training_render_calls=0,diagnostic_render_calls=0,backward_calls=0,optimizer_updates=0)
    trace_path=NEW/'sampling_traces'/f'{name}.jsonl';current_table=initial
    with trace_path.open('x',buffering=1) as trace:
        for step in range(1,2001):
            tick=time.monotonic()
            if not stack:stack=list(all_uids)
            base_uid=stack.pop(base_sampler.randrange(len(stack)));assert base_uid==base_sequence[step-1]
            selected=selector.choose(step);i=by_uid[selected['uid']];cam=cams[i]
            for optimizer in [model.optimizer,deform.optimizer]:
                for group in optimizer.param_groups:group['lr']=lrs[step-1][group['name']]
            rng_digest=hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()
            warp=deform.step(model.get_xyz.detach(),cam.fid.unsqueeze(0).expand(n,-1));rgb=render(cam,model,pipe,background,warp['d_xyz'],warp['d_rotation'],warp['d_scaling'])['render']
            alpha=render(cam,model,pipe,torch.zeros_like(background),warp['d_xyz'],warp['d_rotation'],warp['d_scaling'],override_color=torch.ones_like(model.get_xyz))['render'][0]
            rgb_loss=.8*(rgb-cam.original_image).abs().mean()+.2*(1-ssim(rgb,cam.original_image));alpha_loss=old_alpha(alpha,alphas[i].float()/255);reg=warp['reg_spatial'].mean()*opt.lambda_spatial_tv+warp['reg_temporal'].mean()*opt.lambda_temporal_tv
            loss=rgb_loss+.1*alpha_loss+reg;assert bool(torch.isfinite(loss));loss.backward();model.optimizer.step();deform.optimizer.step();model.optimizer.zero_grad(set_to_none=True);deform.optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize();counts['training_loop_seconds']+=time.monotonic()-tick;counts['training_render_calls']+=2;counts['backward_calls']+=1;counts['optimizer_updates']+=1;assert len(model.get_xyz)==n and model.active_sh_degree==degree
            record=dict(step=step,global_actual_updates=base+step,**selected,loss=float(loss),rgb_loss=float(rgb_loss),alpha_loss=float(alpha_loss),regularization=float(reg),learning_rates=lrs[step-1],perturbation_rng_before_sha256=rng_digest,points=n,active_opacity_gt0005=int((model.get_opacity>.005).sum()),demand_version=0 if step<=1000 else 1000,wall_seconds=time.monotonic()-loop);trace.write(json.dumps(record,allow_nan=False)+'\n')
            if step==1000:
                current_table=NEW/'demand_tables'/f'{name}_1000.json';table=score(model,deform,pipe,data,args.label,current_table,selector.exposure);selector.update(table);counts['diagnostic_seconds']+=table['seconds'];counts['diagnostic_render_calls']+=table['render_calls']
            if step%100==0 or step==1:
                assert shutil.disk_usage(NEW).free>=35*2**30;save(run/'progress.json',dict(state='training',pid=os.getpid(),device=device,**record,costs=counts))
            if step in [500,1000,2000]:
                for o in [model.optimizer,deform.optimizer]:
                    for g in o.param_groups:assert [float(o.state[p]['step']) for p in g['params']]==[v+step for v in original_steps[g['name']]]
                counts['snapshot_seconds']+=snapshot(run,base,step,model,deform,pipe,data,stack,counts,source,selector,base_sampler,current_table)
                save(run/'budget'/f'{base+step}.json',dict(actual_updates=base+step,extra_updates=step,costs=counts,entry_seconds_so_far=time.monotonic()-begin))
    assert stack==temp_stack and base_sampler.getstate()==generator.getstate()
    save(run/'exposure.json',dict(observations=[dict(uid=r['uid'],view=r['view'],frame=r['frame'],bin=r['bin'],count=int(selector.exposure[i]),areas=r['areas']) for i,r in enumerate(candidate['observations'])],total=2000))
    save(run/'complete.json',dict(state='optimization_complete_replay_pending',label=args.label,arm=args.arm,anchor=args.anchor,extra_updates=2000,global_actual_updates=base+2000,points=n,device=device,costs=counts,initial_score_shared_seconds=json.loads(initial.read_text())['seconds'],initial_score_cost_scope='shared preparation,do not multiply byfour;U/P production would not need diagnostic refresh',end_to_end_seconds=time.monotonic()-begin,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),sampling_trace=str(trace_path),sampling_sha256=sha(trace_path),base_sequence_sha256=sha(run/'base_sequence.json'),lr_sha256=sha(run/'learning_rates.json')))
    print(name,'complete',base+2000,flush=True)
if __name__=='__main__':main()
