"""Portable asset packing and baked replay; no original image access on load."""
from runtime import *

@torch.no_grad()
def export_asset(out,g,d,q,config,data,step,inputs):
    out=Path(out);out.mkdir(exist_ok=False,parents=True)
    g.save_ply(str(out/'canonical.ply'));atomic_torch(out/'base.pt',dict(hash=d.grid4d.state_dict(),deform=d.deform.state_dict()))
    if q is not None:
        z=bake(q,d,data,inputs);atomic_torch(out/'query.pt',dict(arm=q.arm,base_dim=q.base_dim,state=q.state_dict(),Z=z))
    save(out/'config.json',config);save(out/'cameras.json',data)
    save(out/'asset.json',dict(actual_step=step,active_sh_degree=g.active_sh_degree,points=len(g.get_xyz),arm='R0' if q is None else q.arm,continuous_time='linear adjacent integer Z,decode with native feature at actual t',scope='current recorded motion;no arbitrary action or world-scale claim',files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))

def load_portable(out):
    out=Path(out);meta=json.loads((out/'asset.json').read_text());cfg=json.loads((out/'config.json').read_text());data=json.loads((out/'cameras.json').read_text())
    for name,h in meta['files'].items():assert sha(out/name)==h
    g=GaussianModel(cfg['dataset']['sh_degree']);g.load_ply(str(out/'canonical.ply'));g.active_sh_degree=meta['active_sh_degree']
    d=new_deform(cfg);state=torch.load(out/'base.pt',map_location='cuda');d.grid4d.load_state_dict(state['hash'],strict=True);d.deform.load_state_dict(state['deform'],strict=True)
    q=z=None
    if meta['arm']!='R0':
        a=torch.load(out/'query.pt',map_location='cuda');s=a['state'];q=EVQ(s['anchors'],s['center'],float(s['length']),a['base_dim'],a['arm']);q.load_state_dict(s,strict=True);q.eval();z=a['Z'];assert z.dtype==torch.float32 and z.shape==(data['frame_count'],128,64)
    return g,d,q,z,cfg,data,meta

def no_source_reads(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        p=Path(os.fsdecode(args[0])).absolute()
        assert not p.is_relative_to(BASE),'Independent replay cannot read Random8: '+str(p)

@torch.no_grad()
def expectations(path,g,d,q,config,data,step,inputs):
    frames=[0.,12.25,30.5,float(data['frame_count']-1)];records={};pipe=SimpleNamespace(**config['pipeline'])
    for frame in frames:
        state=geometry(g,d,q,data,frame,step,inputs)
        for v in [0,8,16]:
            c=camera(data,v,frame)
            for bg in [0.,.5,1.]:
                records[f'{frame}_{v}_{bg}']=raster(g,pipe,c,state,bg)['render'].cpu().numpy()
            records[f'{frame}_{v}_alpha']=raster(g,pipe,c,state,0.,True)['render'][:1].cpu().numpy()
    np.savez_compressed(path,**records)

@torch.no_grad()
def independent_validate(out,expected):
    sys.addaudithook(no_source_reads);device=verify();g,d,q,z,cfg,data,meta=load_portable(out);values=np.load(expected);pipe=SimpleNamespace(**cfg['pipeline']);rows=[]
    for f in [0.,12.25,30.5,float(data['frame_count']-1)]:
        state=geometry(g,d,q,data,f,meta['actual_step'],baked=z)
        for v in [0,8,16]:
            cam=camera(data,v,f);alpha=raster(g,pipe,cam,state,0.,True)['render'][:1];black=raster(g,pipe,cam,state,0.)['render']
            ae=float((alpha-torch.as_tensor(values[f'{f}_{v}_alpha'],device='cuda')).abs().max());assert ae<=1e-5
            for bg in [0.,.5,1.]:
                rgb=raster(g,pipe,cam,state,bg)['render'];error=float((rgb-torch.as_tensor(values[f'{f}_{v}_{bg}'],device='cuda')).abs().max());composition=float((rgb-(black+bg*(1-alpha))).abs().max());assert max(error,composition)<=1e-5,(f,v,bg,error,composition)
                rows.append(dict(frame=f,view=v,background=bg,max_abs=error,composition_max_abs=composition,alpha_max_abs=ae))
    result=dict(passed=True,device=device,original_Random8_read_guard_active=True,asset=str(out),same_3d_state_reused_for_all_cameras=True,fractional_definition_fixed=True,tolerance=1e-5,rows=rows);save(Path(out)/'replay_validation.json',result);return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--asset',required=True);p.add_argument('--expected',required=True);a=p.parse_args();independent_validate(Path(a.asset),Path(a.expected));print('independent replay passed',flush=True)
