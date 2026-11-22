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
