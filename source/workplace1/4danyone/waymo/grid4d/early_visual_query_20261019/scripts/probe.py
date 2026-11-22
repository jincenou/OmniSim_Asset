"""Real-rasterizer mechanism test; <=200 updates, excluded from formal curves."""
from train import *
import gc

def main():
    sys.addaudithook(guard_training);device=verify();out=NEW/'research/probe';out.mkdir(exist_ok=False)
    g,d,q,qo,cfg,data=cold('N11','R3');inp=Inputs('N11',data,all_frames=False);pipe=SimpleNamespace(**cfg['pipeline']);cam,_,_=inp.target(1,12)
    with torch.no_grad():
        state=geometry(g,d,q,data,12,4000,inp);w=state['warp'];native=render(cam,g,pipe,torch.full((3,),.5,device='cuda'),w['d_xyz'],w['d_rotation'],w['d_scaling'])
        native_diff=float((native['render']-raster(g,pipe,cam,state)['render']).abs().max());assert native_diff<=1e-6
        world_diff=float((native['world_means']-state['means']).abs().max());assert world_diff==0
    initial={k:v.detach().clone() for k,v in q.named_parameters()};rows=[]
    for i in range(12):rows.append(update(g,d,q,qo,cfg,data,inp,12,[1,7],4001+i,structure=False))
    required=['query.weight','key.weight','value.weight','offset.weight','head.weight']
    evidence={k:dict(gradient=max(r['query_grad_abs_sum'][k] for r in rows[1:]),change=float((dict(q.named_parameters())[k]-initial[k]).abs().max())) for k in required}
    assert all(v['gradient']>0 and v['change']>0 for v in evidence.values()),evidence
    with torch.no_grad():
        t=torch.tensor([[.1]],device='cuda');z,ed=q.encode(d,t,inp.support(12),inp.projections)
        corrupt=[x.clone() for x in inp.support(12)]
        for x in corrupt:x[:,:3]=1-x[:,:3]
        zp,_=q.encode(d,t,corrupt,inp.projections);support_response=float((zp-z).abs().max());assert support_response>1e-7
        w,feat=base_features(d,g.get_xyz.detach(),t);p,col,reg=q.decode(g,t,z,feat,.01);p2,col2,reg2=q.decode(g,t,z,feat,1.)
        assert torch.equal(reg['residual_regularizer'],reg2['residual_regularizer']);assert torch.allclose(p*100,p2,atol=1e-7)
        dp,dc,_=q.decode(g,t,zp,feat,1.);residual_response=float((dc-col2).abs().max());assert residual_response>1e-9
        behind=inp.projections.clone();behind[:,:,-1]=0;behind[:,3,-1]=-1
        zn,nd=q.encode(d,t,inp.support(12),behind);assert torch.isfinite(zn).all() and float(nd['null_mean'])==1.
        g.densify_and_clone(torch.ones(len(g.get_xyz),1,device='cuda'),0.,1000.)
        mask=torch.zeros(len(g.get_xyz),dtype=torch.bool,device='cuda');mask[::7]=True;g.prune_points(mask)
        _,ft=base_features(d,g.get_xyz.detach(),t);_,_,top=q.decode(g,t,z,ft);assert top['association'].shape==(len(g.get_xyz),4);assert int(top['association'].max())<128
    rows.append(update(g,d,q,qo,cfg,data,inp,12,[1,7],4013,structure=False))
    s=checkpoint(g,d,q,qo,cfg,data,4013,{'technical':True});atomic_torch(out/'recovery.pt',s);del s;torch.cuda.empty_cache()
    s=torch.load(out/'recovery.pt',map_location='cpu');restore(s,g,d,q,qo,cfg);del s
    with torch.no_grad():
        table=torch.stack([q.latent_at(d,f,data['frame_count'],inp.support,inp.projections) for f in [12,13]])
        online=q.latent_at(d,12.25,data['frame_count'],inp.support,inp.projections);assert torch.equal(online,table[0]*.75+table[1]*.25)
        finalstate=geometry(g,d,q,data,12.25,4013,inp);fulltable=torch.zeros(data['frame_count'],128,64,device='cuda');fulltable[12:14]=table
        bakedstate=geometry(g,d,q,data,12.25,4013,baked=fulltable);equal(finalstate['means'],bakedstate['means']);equal(finalstate['sh'],bakedstate['sh'])
    report=dict(passed=True,technical_updates=len(rows),device=device,native_RGB_max_abs=native_diff,native_world_max_abs=world_diff,gradient_and_parameter_updates=evidence,support_latent_response=support_response,support_residual_response=residual_response,pre_gate_regularizer_verified=True,null_all_invalid_weight=float(nd['null_mean']),topology_after_clone_prune_points=len(g.get_xyz),full_state_restore_exact=True,continuous_bake_exact=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),diagnostics=rows[-1]['diagnostics'],formal_training=False)
    save(out/'report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
