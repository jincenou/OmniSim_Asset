from assets import *
import argparse

def load_recovery(path):
    s=torch.load(path,map_location='cpu');cfg=s['config'];data=s['data'];g=GaussianModel(cfg['dataset']['sh_degree'])
    for k in PARAMETERS:setattr(g,k,torch.nn.Parameter(s['gaussians'][k].cuda(),requires_grad=False))
    g.active_sh_degree=s['active_sh_degree'];d=new_deform(cfg);d.grid4d.load_state_dict(s['hash'],strict=True);d.deform.load_state_dict(s['deform'],strict=True)
    q=None
    if s['query'] is not None:
        a={k:v.cuda() for k,v in s['query'].items()};q=DirectQuery(a['center'],float(a['length']),d.spatial_dim+d.temporal_dim,cfg['evq_arm']);q.load_state_dict(a,strict=True);q.eval()
    return g,d,q,cfg,data,s['step']

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--key',required=True);args=ap.parse_args();verify();run=NEW/'runs'/args.key;assert (run/'training.complete.json').exists()
    g,d,q,cfg,data,step=load_recovery(run/'recovery_32999.pt');inp=Inputs(args.key[:3],data,False);out=NEW/'assets'/args.key
    export_asset(out,g,d,q,cfg,data,step,inp);expectations(run/'replay_expected.npz',g,d,q,cfg,data,step,inp)
    # Exact export tensor identity, apart from PLY's explicit native activations
    # (native PLY stores raw SH/logscale/logit/quaternion as float32).
    g2,d2,q2,z2,c2,data2,meta=load_portable(out)
    for k in PARAMETERS:equal(getattr(g,k),getattr(g2,k))
    equal(d.grid4d.state_dict(),d2.grid4d.state_dict());equal(d.deform.state_dict(),d2.deform.state_dict())
    if q is not None:equal(q.state_dict(),q2.state_dict())
    save(out/'export_validation.json',dict(passed=True,recovery_sha256=sha(run/'recovery_32999.pt'),raw_PLY_parameters_exact=True,base_hash_deform_exact=True,query_exact=q is not None,parameter_counts=dict(gaussian=sum(getattr(g,k).numel() for k in PARAMETERS),base_hash=sum(p.numel() for p in d.grid4d.parameters()),base_deform=sum(p.numel() for p in d.deform.parameters()),new_query=0 if q is None else sum(p.numel() for p in q.parameters())),asset_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file())))
    print(args.key,'asset exported; new-process replay still required',flush=True)
if __name__=='__main__':main()
