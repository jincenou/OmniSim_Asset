"""Technical checks only; probe states are never eligible experiment assets."""
from update import *
import gc


def main():
    sys.addaudithook(guard_training)
    device = verify()
    rows = []
    shared = {}
    initial_render = None
    old = ROOT/'early_visual_query_20261019'
    for arm in ('P1', 'P2', 'P3', 'P4'):
        g,d,q,qo,cfg,data = cold('N06',arm)
        source = old/'runs/N06_warmup/recovery_2999.pt'
        state = torch.load(source,map_location='cpu')
        restore(state,g,d,q,qo,cfg,allow_cold=True)
        del state
        for name,value in q.state_dict().items():
            if name in shared: equal(shared[name],value)
            else: shared[name]=value.detach().cpu().clone()
        inp=Inputs('N06',data,False)
        pipe=SimpleNamespace(**cfg['pipeline'])
        with torch.no_grad():
            s=geometry(g,d,q,data,30,4000,inp)
            assert torch.count_nonzero(s['delta'])==0
            cam,_,_=inp.target(1,30)
            image=raster(g,pipe,cam,s)['render'].cpu()
            if initial_render is None: initial_render=image
            else: assert torch.equal(initial_render,image)
        # Head updates first, then evidence/offset gradients can propagate.
        traces=[update(g,d,q,qo,cfg,data,inp,30,[1,7],step,structure=False) for step in (4001,4002,4003)]
        assert traces[-1]['query_grad_abs_sum']['head.weight']>0
        if arm in ('P3','P4'):
            for key in ('query.weight','key.weight','offset.weight','image_projection.weight'):
                assert traces[-1]['query_grad_abs_sum'][key]>0,(arm,key)
        assert all(torch.isfinite(p).all() for p in q.parameters())
        t=g.get_xyz.new_tensor([[.5]])
        warp,feat=base_features(d,g.get_xyz.detach(),t,False)
        base=native_world(g.get_xyz,warp['d_xyz'])
        pyramids=inp.support(30)
        depth=depth_moments(g,warp,base,inp)
        assert torch.isfinite(depth).all() and depth[:,1].min()>=-1e-6 and depth[:,1].max()<=1.0001
        # A small fixed selection verifies exact chunking and gradient wiring.
        ix=torch.arange(29,device='cuda')
        small=SimpleNamespace(get_xyz=g.get_xyz[ix].detach(),_features_dc=g._features_dc[ix].detach(),_scaling=g._scaling[ix].detach(),_rotation=g._rotation[ix].detach(),_opacity=g._opacity[ix].detach())
        ff=feat[ix].detach();ww=base[ix].detach();pp=inp.projections
        outputs=[];gradients=[]
        for chunk in (29,7):
            q.zero_grad(set_to_none=True);q.chunk=chunk
            delta,reg,info=q(small,ff,ww,t,pyramids,pp,depth)
            (delta.square().mean()+1e-4*reg).backward()
            outputs.append(delta.detach().clone())
            gradients.append({k:p.grad.detach().clone() for k,p in q.named_parameters() if p.grad is not None})
        output_error=float((outputs[0]-outputs[1]).abs().max())
        assert output_error<1e-6
        grad_error=max(float((gradients[0][k]-gradients[1][k]).abs().max()) for k in gradients[0])
        assert grad_error<1e-5
        # Reordering the Gaussian rows must reorder predictions identically.
        with torch.no_grad():
            reverse=torch.arange(28,-1,-1,device='cuda')
            reordered=SimpleNamespace(**{k:v[reverse] for k,v in vars(small).items()})
            ordered=q(reordered,ff[reverse],ww[reverse],t,pyramids,pp,depth)[0]
            assert torch.allclose(ordered,outputs[1][reverse],atol=1e-6,rtol=1e-5)
        rows.append(dict(arm=arm,zero_initial_render_exact=True,matched_shared_parameters=True,restore_exact=True,chunk_max_abs=output_error,chunk_gradient_max_abs=grad_error,row_permutation_passed=True,gradients=traces[-1]['query_grad_abs_sum'],peak_allocated=int(torch.cuda.max_memory_allocated()),parameter_count=sum(p.numel() for p in q.parameters())))
        del g,d,q,qo,inp,small,feat,base,warp,depth,outputs,gradients
        gc.collect();torch.cuda.empty_cache()
    # Synthetic occlusion condition: same P3/P4 weights, neutral depth must match.
    g,d,q,qo,cfg,data=cold('N06','P4');inp=Inputs('N06',data,False)
    t=g.get_xyz.new_tensor([[.5]]);warp,feat=base_features(d,g.get_xyz.detach(),t,False)
    world=native_world(g.get_xyz,warp['d_xyz'])[:16];context=torch.randn(16,64,device='cuda')
    p0,p1=inp.support(30);neutral=torch.zeros(4,3,640,352,device='cuda')
    q.arm='P3';a,_=q.sample(context,world,p0,p1,inp.projections,neutral)
    q.arm='P4';b,neutral_info=q.sample(context,world,p0,p1,inp.projections,neutral)
    assert torch.equal(a,b) and float(neutral_info[3])==1.
    occluder=neutral.clone();occluder[:,0]=.1;occluder[:,1]=1;occluder[:,2]=.01
    _,blocked_info=q.sample(context,world,p0,p1,inp.projections,occluder)
    assert float(blocked_info[3])<1.,float(blocked_info[3])
    save(NEW/'research/preflight.json',dict(passed=True,device=device,technical_updates=12,not_formal_states=True,rows=rows,neutral_visibility_exact=True,synthetic_occluder_downweight=float(blocked_info[3]),limitations='Finite technical checks, not proof of physical correspondence or all renderer gradients'))
    print('preflight passed',flush=True)


if __name__=='__main__':main()
