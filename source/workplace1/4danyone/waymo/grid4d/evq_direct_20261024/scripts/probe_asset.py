"""Large real geometry backward and isolated replay before formal training."""
from export import *
from update import update


def main():
    device=verify();old=ROOT/'early_visual_query_20261019'
    g,d,q,qo,cfg,data=cold('N04','P4')
    source=old/'runs/N04_R0/recovery_32999.pt'
    state=torch.load(source,map_location='cpu')
    for key in PARAMETERS:setattr(g,key,torch.nn.Parameter(state['gaussians'][key].cuda()))
    g.training_setup(SimpleNamespace(**cfg['optimization']))
    g.active_sh_degree=state['active_sh_degree']
    for k,v in state['statistics'].items():setattr(g,k,v.cuda())
    d.grid4d.load_state_dict(state['hash'],strict=True);d.deform.load_state_dict(state['deform'],strict=True)
    del state
    cfg['evq_arm']='P4';cfg['direct_arm']='P4'
    inp=Inputs('N04',data,True);torch.cuda.reset_peak_memory_stats()
    began=time.monotonic()
    traces=[update(g,d,q,qo,cfg,data,inp,30,[1,7],step,structure=False) for step in (18001,18002)]
    seconds=time.monotonic()-began
    save(NEW/'research/capacity_probe.json',dict(passed=True,device=device,points=len(g.get_xyz),technical_updates=2,not_formal=True,source_geometry=str(source),peak_allocated=int(torch.cuda.max_memory_allocated()),peak_reserved=int(torch.cuda.max_memory_reserved()),two_updates_seconds=seconds,final_loss=traces[-1]['loss']))
    folder=NEW/'tmp/probe_asset'
    export_asset(folder,g,d,q,cfg,data,32999,inp)
    expectations(NEW/'tmp/probe_expected.npz',g,d,q,cfg,data,32999,inp)
    g2,d2,q2,z2,c2,data2,meta=load_portable(folder)
    for key in PARAMETERS:equal(getattr(g,key),getattr(g2,key))
    equal(q.state_dict(),q2.state_dict())
    assert z2.shape==(data['frame_count'],len(g.get_xyz),6)
    print('capacity/export passed; isolated replay pending',flush=True)


if __name__=='__main__':main()
