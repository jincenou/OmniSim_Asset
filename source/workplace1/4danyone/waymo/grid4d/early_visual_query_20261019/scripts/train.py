"""Cold/continued paired-view training; all arms retain native structural ordering."""
from runtime import *
from utils.loss_utils import l1_loss,ssim
import argparse,shutil

def update(g,d,q,qo,config,data,inputs,frame,views,step,structure=True,observe=True):
    opt=SimpleNamespace(**config['optimization']);pipe=SimpleNamespace(**config['pipeline'])
    if step%1000==0:g.oneupSHdegree()
    state=geometry(g,d,q,data,frame,step,inputs,regularize=step>opt.densify_until_iter)
    packages=[];loss=state['reg'];aw=.1*min(1.,max(0.,(step-15000)/3000))
    for view in views:
        cam,rgb,refalpha=inputs.target(view,frame);pkg=raster(g,pipe,cam,state)
        photometric=(1-opt.lambda_dssim)*l1_loss(pkg['render'],rgb)+opt.lambda_dssim*(1-ssim(pkg['render'],rgb))
        if aw>0:
            ap=raster(g,pipe,cam,state,0.,True)['render'][:1];mask=refalpha>.5
            assert bool(mask.any()) and bool((~mask).any())
            photometric=photometric+aw*.5*((ap-refalpha).abs()[mask].mean()+(ap-refalpha).abs()[~mask].mean())
        loss=loss+photometric/2;packages.append(pkg)
    assert torch.isfinite(loss);loss.backward()
    grads={} if q is None or not observe else {k:float(v.grad.abs().sum()) if v.grad is not None else 0. for k,v in q.named_parameters()}
    event=None
    with torch.no_grad():
        for p in packages:
            visible=p['visibility_filter'];g.max_radii2D[visible]=torch.maximum(g.max_radii2D[visible],p['radii'][visible])
        if structure and step<opt.densify_until_iter:
            # Loss averaged over two images; restore per-image gradient scale and
            # count BOTH exposures in denominator. Threshold stays native .0002.
            for p in packages:
                p['viewspace_points'].grad.mul_(2);g.add_densification_stats(p['viewspace_points'],p['visibility_filter'])
            if step>opt.densify_from_iter and step%opt.densification_interval==0:
                centers=np.asarray([c['camtoworld'] for c in data['cameras'] if c['view'] in data['train_views']])[:,:3,3]
                extent=float(np.linalg.norm(centers-centers.mean(0),axis=1).max()*1.1);before=len(g.get_xyz)
                g.densify_and_prune(opt.densify_grad_threshold,.005,extent,20 if step>opt.opacity_reset_interval else None,opt.disable_ws_prune)
                event=dict(step=step,before=before,after=len(g.get_xyz),operation='native_clone_split_prune')
            if step%opt.opacity_reset_interval==0:g.reset_opacity()
        # Preserve upstream ordering. Replaced Gaussian Parameter rows have no
        # gradient on structural iterations; expose this instead of inventing steps.
        groups_updated={group['name']:any(p.grad is not None for p in group['params']) for group in g.optimizer.param_groups}
        g.optimizer.step();g.update_learning_rate(step);d.optimizer.step();d.update_learning_rate(step)
        if qo is not None and step>=3000:
            lr=5e-5+.5*(5e-4-5e-5)*(1+math.cos(math.pi*(step-3000)/29999))
            for group in qo.param_groups:group['lr']=lr
            qo.step();qo.zero_grad(set_to_none=True)
        g.optimizer.zero_grad(set_to_none=True);d.optimizer.zero_grad(set_to_none=True)
    return dict(loss=float(loss.detach()),points=len(g.get_xyz),diagnostics={k:float(v.detach()) for k,v in state['diag'].items()} if observe else {},query_grad_abs_sum=grads,gaussian_groups_updated=groups_updated,structure=event)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--label',required=True);ap.add_argument('--arm',choices=['R0','R1','R2','R3'],required=True);ap.add_argument('--warmup',action='store_true');ap.add_argument('--resume');args=ap.parse_args()
    assert (NEW/'PLAN.json').exists() and (NEW/'research/preflight.json').exists()
    assert json.loads((NEW/'research/preflight.json').read_text())['passed']
    plan=specification();assert plan['state']=='frozen_before_training'
    for p,h in plan['implementation_sha256'].items():assert sha(NEW/p)==h,p
    sys.addaudithook(guard_training);device=verify();label=args.label;arm=args.arm
    run=NEW/'runs'/f'{label}_{"warmup" if args.warmup else arm}'
    if args.resume:assert run.is_dir()
    else:run.mkdir(exist_ok=False)
    start=time.monotonic();g,d,q,qo,config,data=cold(label,arm);config['evq_arm']=arm;config['dataset']['model_path']=str(run)
    inp=Inputs(label,data);step0=0;counts=dict(updates=0,target_renders=0,alpha_renders=0,support_times=0,gaussian_updates_by_group={});elapsed_previous=0.
    source=Path(args.resume) if args.resume else NEW/'runs'/f'{label}_warmup'/'recovery_2999.pt'
    if not args.warmup or args.resume:
        state=torch.load(source,map_location='cpu');step0=restore(state,g,d,q,qo,config,allow_cold=not args.resume)
        counts=state['extra']['counts'];elapsed_previous=state['extra']['training_wall_seconds'];del state
        save(run/'resume.json',dict(source=str(source),sha256=sha(source),actual_step=step0,exact_tensors_optimizers_rng=True))
    save(run/'config.json',config);save(run/'cameras.json',data);save(run/'device.json',device)
    schedule=json.loads((NEW/'data'/label/'schedule.json').read_text());limit=2999 if args.warmup else 32999
    loaded=time.monotonic();trace=(run/'trace.jsonl').open('a' if args.resume else 'x',buffering=1)
    for step in range(step0+1,limit+1):
        if step%100==1:assert shutil.disk_usage(NEW).free>plan['resource']['minimum_disk_free_GiB']*2**30,'Disk reserve reached; resume from saved state after space is available'
        frame,v1,v2=schedule[step-1];r=update(g,d,q,qo,config,data,inp,frame,[v1,v2],step,observe=step%100==0 or step in [1,2999,3000,3001,3999,5000,15000,18000,32999])
        counts['updates']+=1;counts['target_renders']+=2;counts['alpha_renders']+=2*int(step>15000);counts['support_times']+=int(q is not None and arm in ['R2','R3'] and step>=3000)
        for k,v in r['gaussian_groups_updated'].items():counts['gaussian_updates_by_group'][k]=counts['gaussian_updates_by_group'].get(k,0)+int(v)
        if step%100==0 or step in [1,2999,3000,3001,3999,5000,15000,18000,32999] or r['structure']:
            row=dict(step=step,frame=frame,views=[v1,v2],wall_seconds=time.monotonic()-loaded,**r);trace.write(json.dumps(row,allow_nan=False)+'\n');save(run/'progress.json',dict(step=step,total=limit,points=len(g.get_xyz),at=stamp(),device=device,state='training'));print(label,arm,step,r['loss'],r['points'],flush=True)
        if step in plan['budget']['checkpoints'] or step%2000==0:
            extra=dict(counts=counts,training_wall_seconds=elapsed_previous+time.monotonic()-loaded,load_seconds=loaded-start,plan_sha256=sha(NEW/'PLAN.json'),schedule_sha256=sha(NEW/'data'/label/'schedule.json'),arm=arm,label=label,device=device,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
            target=run/(f'recovery_{step}.pt' if step in plan['budget']['checkpoints'] else 'recovery_latest.pt');atomic_torch(target,checkpoint(g,d,q,qo,config,data,step,extra))
            if step in plan['budget']['checkpoints']:
                save(run/f'snapshot_{step}.json',dict(actual_updates=step,active_sh_degree=g.active_sh_degree,points=len(g.get_xyz),recovery_sha256=sha(target),**extra))
    assert counts['updates']==limit;trace.close();save(run/'training.complete.json',dict(step=limit,at=stamp(),counts=counts,wall_seconds=time.monotonic()-start))

if __name__=='__main__':main()
