"""Only new-entry tests, isolated updates <=200; no quality-based hyperparameter choice."""
from update import *
from assets import *
import copy,argparse
OLD=ROOT/'early_visual_query_20261019'

def frozen_initial(label,variant):
    g,d,q,qo,cfg,data=cold(label,variant);s=torch.load(OLD/'runs'/f'{label}_warmup/recovery_2999.pt',map_location='cpu');restore(s,g,d,q,qo,cfg,True);del s;cfg['primitive_variant']=variant;cfg['evq_arm']='R3';return g,d,q,qo,cfg,data

@torch.no_grad()
def mature(label,variant):
    asset=OLD/'assets'/f'{label}_R3';cfg=json.loads((asset/'config.json').read_text());data=json.loads((asset/'cameras.json').read_text());meta=json.loads((asset/'asset.json').read_text());g=GaussianModel(cfg['dataset']['sh_degree']);g.load_ply(str(asset/'canonical.ply'));g.active_sh_degree=meta['active_sh_degree'];d=new_deform(cfg);b=torch.load(asset/'base.pt',map_location='cuda');d.grid4d.load_state_dict(b['hash']);d.deform.load_state_dict(b['deform']);a=torch.load(asset/'query.pt',map_location='cuda');s=a['state'];q=EVQ(s['anchors'],s['center'],float(s['length']),a['base_dim'],'R3',variant=variant);q.load_state_dict(s);return g,d,q,cfg,data,a['Z']

def chunk_check(q,g,d,data,inp):
    # Same full derivative path on a bounded actual-geometry technical subset;
    # formal training never subsamples points.
    t=g.get_xyz.new_tensor([[22/data['time_denominator']]])
    with torch.no_grad():
        w,feat=base_features(d,g.get_xyz.detach(),t,False);base=native_world(g.get_xyz,w['d_xyz']);z,ed,ctx=q.encode_geo(d,t,inp.support(22),inp.projections);dp,dc,dd=q.decode(g,t,z,feat,1.);pc=q.bind_context(g,ctx,dd['association']);ix=torch.linspace(0,len(base)-1,257,device='cuda').long();world=(base+dp)[ix].detach();context=pc[ix].detach();x=q.normalized(g.get_xyz)[ix].detach();features=feat[ix].detach();p0,p1=inp.support(22)
    records=[];outputs=[];grads=[]
    for chunk,check in [(257,False),(64,False),(64,True)]:
        q.zero_grad(set_to_none=True);W=world.clone().requires_grad_();C=context.clone().requires_grad_();vals=[]
        from torch.utils.checkpoint import checkpoint as activation_ckpt
        for i in range(0,257,chunk):
            args=(W[i:i+chunk],C[i:i+chunk],x[i:i+chunk],features[i:i+chunk],t,p0,p1,inp.projections)
            c,_=activation_ckpt(q.local_color,*args,use_reentrant=False) if check else q.local_color(*args);vals.append(c)
        out=torch.cat(vals);loss=out.square().mean()+out.mean()*.013;loss.backward();outputs.append(out.detach());grad={k:v.grad.detach().clone() for k,v in q.named_parameters() if v.grad is not None};grad['world']=W.grad.detach().clone();grad['context_input']=C.grad.detach().clone();grads.append(grad)
        records.append(dict(chunk=chunk,activation_checkpoint=check,world_gradient_abs=float(W.grad.abs().sum()),parameters_with_grad=len(grad)))
    for i in [1,2]:
        assert float((outputs[i]-outputs[0]).abs().max())<=1e-5;assert grads[i].keys()==grads[0].keys()
        for k in grads[0]:
            err=float((grads[i][k]-grads[0][k]).abs().max());tol=1e-5+1e-3*float(grads[0][k].abs().max());assert err<=tol,(k,err,tol)
    assert records[0]['world_gradient_abs']>0
    with torch.no_grad():
        lat,summary,tr=q.local_latent(world,context,p0,p1,inp.projections,True)
        assert tr['grids'].shape==(4,257,2,4,2) and tr['weights'].shape==(257,2,33)
        # Pixel formula and fixed cross plus learned bounded offset are literal.
        for level,scale in enumerate([2,4]):
            pixel=(tr['grids'][:,:,level].double()+1)*world.new_tensor([352,640]).double()/2-.5;center=(tr['ndc'].double()+1)*world.new_tensor([352,640]).double()/2-.5
            expected=center[:,:,None]+(q.cross[None,None].double()+tr['offsets'][:,:,level].double())*scale
            valid=tr['valid'][:,:32].reshape(257,4,2,4).permute(1,0,2,3)[:,:,level]
            assert float((pixel-expected).abs()[valid].max())<1e-4
        behind=inp.projections.clone();behind[:,:,-1]=0;behind[:,3,-1]=-1
        _,ns,nt=q.local_latent(world,context,p0,p1,behind,True);assert torch.isfinite(ns).all() and float(ns[0])==1 and float(ns[1])==0
    return dict(passed=True,records=records,null_all_behind=1.,projection_footprint=True,forward_tolerance=1e-5,gradient_tolerance='maxabs<=1e-5+1e-3*maxabs(reference)',not_renderer_global_finite_difference=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--label',required=True);a=ap.parse_args();label=a.label;device=verify();out=NEW/'research'/f'preflight_{label}';out.mkdir(exist_ok=False);torch.set_num_threads(2);report=dict(device=device,technical_updates=0,checks={});begin=time.monotonic()
    # Cold RNG and identity across variants.
    initial=[]
    for variant in ['G0','G1','G2']:
        g,d,q,qo,cfg,data=frozen_initial(label,variant);initial.append({k:v.detach().cpu().clone() for k,v in q.state_dict().items()});assert len(list(q.parameters()))==len(list(q.named_parameters(remove_duplicate=False)));assert len({id(p) for group in qo.param_groups for p in group['params']})==len(list(q.parameters()));del g,d,q,qo
    equal(initial[0],initial[1]);equal(initial[1],initial[2]);del initial;report['checks']['same_cold_initialization_no_duplicate_parameters']=True
    # Technical mature weights are used only for robust nonzero Jacobian test, never training initialization.
    g,d,q,cfg,data,Z=mature(label,'G0');inp=Inputs(label,data,False);pipe=SimpleNamespace(**cfg['pipeline'])
    with torch.no_grad():
        f=22;t=g.get_xyz.new_tensor([[f/data['time_denominator']]]);warp,feat=base_features(d,g.get_xyz.detach(),t,False);base=native_world(g.get_xyz,warp['d_xyz']);dp,dc,_=q.decode(g,t,Z[f],feat,1.);old=dict(warp=warp,means=base+dp,sh=torch.cat([g.get_features[:,:1]+dc[:,None],g.get_features[:,1:]],1));new=geometry(g,d,q,data,f,32999,inp);cam=camera(data,0,f);err=float((raster(g,pipe,cam,old)['render']-raster(g,pipe,cam,new)['render']).abs().max());assert err<=1e-5
        # DC changes cannot influence genuine accumulated alpha at fixed means.
        changed=dict(new);changed['sh']=new['sh'].clone();changed['sh'][:,:1]+=.2;ae=float((raster(g,pipe,cam,new,0,True)['render']-raster(g,pipe,cam,changed,0,True)['render']).abs().max());assert ae<=1e-5
    report['checks']['G0_old_R3_discrete_max']=err;report['checks']['fixed_geometry_alpha_max']=ae
    q.variant='G2';report['checks']['chunk_G2']=chunk_check(q,g,d,data,inp)
    q.variant='G1';report['checks']['chunk_G1']=chunk_check(q,g,d,data,inp)
    # Actual terminal-scale forward+backward memory/time (no optimizer update).
    q.variant='G2';q.chunk=specification()['model']['local_chunk'];torch.cuda.reset_peak_memory_stats();q.zero_grad(set_to_none=True);torch.cuda.synchronize();tick=time.monotonic();state=geometry(g,d,q,data,22,18001,inp,regularize=True);cam,ref,alpha=inp.target(1,22);cam2,ref2,alpha2=inp.target(7,22);pred=raster(g,pipe,cam,state)['render'];pred2=raster(g,pipe,cam2,state)['render'];apred=raster(g,pipe,cam,state,0,True)['render'];apred2=raster(g,pipe,cam2,state,0,True)['render'];loss=(pred-ref).abs().mean()*.5+(pred2-ref2).abs().mean()*.5+state['reg']+.05*((apred-alpha).abs().mean()+(apred2-alpha2).abs().mean());loss.backward();torch.cuda.synchronize();report['checks']['actual_scale']=dict(points=len(g.get_xyz),seconds_forward_backward=time.monotonic()-tick,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),two_targets=True,all32_tokens=True)
    required=['query.weight','key.weight','value.weight','offset.weight','image_projection.weight','head.weight'];report['checks']['shared_gradients']={k:float(dict(q.named_parameters())[k].grad.abs().sum()) for k in required};assert all(v>0 for v in report['checks']['shared_gradients'].values())
    del g,d,q,state,loss,pred,pred2,apred,apred2,inp;torch.cuda.empty_cache()
    # <=24 updates total across two labels; not for quality tuning.
    g,d,q,qo,cfg,data=frozen_initial(label,'G2');inp=Inputs(label,data,False);before={k:v.detach().clone() for k,v in q.named_parameters()};updates=[]
    for step in range(3999,4011):updates.append(update(g,d,q,qo,cfg,data,inp,12,[1,7],step,structure=False,observe=True));report['technical_updates']+=1
    report['checks']['parameter_changes']={k:float((dict(q.named_parameters())[k]-before[k]).abs().max()) for k in required};assert all(x>0 for x in report['checks']['parameter_changes'].values())
    del before
    with torch.no_grad():
        counts=[len(g.get_xyz)];g.densify_and_clone(torch.ones(len(g.get_xyz),1,device='cuda'),0.,1000.);counts.append(len(g.get_xyz));g.densify_and_split(torch.ones(len(g.get_xyz),1,device='cuda'),0.,.001);counts.append(len(g.get_xyz));mask=torch.zeros(len(g.get_xyz),dtype=torch.bool,device='cuda');mask[::7]=True;g.prune_points(mask);counts.append(len(g.get_xyz));st=geometry(g,d,q,data,12,4010,inp);assert st['means'].shape==(len(g.get_xyz),3) and st['dc_rgb_before_gate'].shape==(len(g.get_xyz),3)
    report['checks']['native_clone_split_prune_rows']=counts
    cfg['primitive_variant']='G2';s=checkpoint(g,d,q,qo,cfg,data,4010,dict(technical=True,counts=report['technical_updates']));atomic_torch(out/'recovery.pt',s);restore(torch.load(out/'recovery.pt',map_location='cpu'),g,d,q,qo,cfg);del s
    with torch.no_grad():
        asset=out/'asset';export_asset(asset,g,d,q,cfg,data,4010,inp);expectations(out/'expected.npz',g,d,q,cfg,data,4010,inp);g2,d2,q2,baked,c2,data2,meta=load_portable(asset)
        for f in [0.,12.25,float(data['frame_count']-1)]:
            online=geometry(g,d,q,data,f,4010,inp);offline=geometry(g2,d2,q2,data,f,4010,baked=baked)
            for name in ['means','sh','dc_rgb_before_gate']:assert float((online[name]-offline[name]).abs().max())<=1e-5
    report.update(passed=True,seconds=time.monotonic()-begin,formal_training=False,old_renderer_finite_difference='Previous global rasterizer FD failure remains; not altered or re-tested here',independent_replay_pending=True)
    save(out/'report.json',report);print(label,'new-entry preflight passed; independent process still required',flush=True)
if __name__=='__main__':main()
