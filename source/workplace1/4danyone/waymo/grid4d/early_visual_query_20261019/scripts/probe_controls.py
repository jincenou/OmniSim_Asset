"""All four controls, warmup topology and common initialized weights, no quality claim."""
from train import *
import gc
def main():
    sys.addaudithook(guard_training);device=verify();rows=[];base=None;shared={};render0=None
    for arm in ['R0','R1','R2','R3']:
        g,d,q,qo,cfg,data=cold('N04',arm);inp=Inputs('N04',data,False);pipe=SimpleNamespace(**cfg['pipeline'])
        current={k:getattr(g,k).detach().cpu() for k in PARAMETERS};current['deform']={k:v.detach().cpu() for k,v in d.deform.state_dict().items()}
        if base is None:base=current
        else:equal(base,current)
        if q is not None:
            for k,v in q.state_dict().items():
                if k in shared:equal(shared[k],v)
                else:shared[k]=v.detach().cpu()
        with torch.no_grad():
            state=geometry(g,d,q,data,12,3000,inp);cam,_,_=inp.target(1,12);im=raster(g,pipe,cam,state)['render'].cpu()
            if render0 is None:render0=im
            else:assert torch.equal(render0,im)
        r=update(g,d,q,qo,cfg,data,inp,12,[1,7],800)
        assert r['structure'] and not any(r['gaussian_groups_updated'].values())
        r2=update(g,d,q,qo,cfg,data,inp,12,[1,7],3000,structure=False)
        assert all(r2['gaussian_groups_updated'].values())
        rows.append(dict(arm=arm,first_zero_residual_identical=True,all_shared_weights_identical=True,warmup_structure=r['structure'],native_replacement_skipped_Gaussian_groups=r['gaussian_groups_updated'],loss=r2['loss']))
        del g,d,q,qo,inp;gc.collect();torch.cuda.empty_cache()
    save(NEW/'research/control_probe.json',dict(passed=True,technical_updates=8,total_technical_updates_to_date=24,device=device,rows=rows));print('all controls passed',flush=True)
if __name__=='__main__':main()
