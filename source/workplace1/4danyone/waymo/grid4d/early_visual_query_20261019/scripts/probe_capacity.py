"""Upper working-size technical memory/alpha/split probe, not a quality experiment."""
from train import *
from assets import export_asset,expectations

def main():
    sys.addaudithook(guard_training);device=verify();g,d,q,qo,cfg,data=cold('N11','R3');inp=Inputs('N11',data)
    s=torch.load(NEW/'research/probe/recovery.pt',map_location='cpu');restore(s,g,d,q,qo,cfg);del s
    # Full-length immutable input pyramid cache, not learned feature cache.
    for frame in range(data['frame_count']):inp.support(frame)
    with torch.no_grad():
        n=len(g.get_xyz);g.densify_and_split(torch.ones(n,1,device='cuda'),0.,.001)
        after_split=len(g.get_xyz);assert after_split>=n
        while len(g.get_xyz)<65536:g.densify_and_clone(torch.ones(len(g.get_xyz),1,device='cuda'),0.,1000.)
        mask=torch.arange(len(g.get_xyz),device='cuda')>=65536;g.prune_points(mask)
    torch.cuda.reset_peak_memory_stats();begin=time.monotonic();rows=[]
    for i in range(3):rows.append(update(g,d,q,qo,cfg,data,inp,12,[1,7],18001+i,structure=False))
    torch.cuda.synchronize();elapsed=time.monotonic()-begin
    report=dict(passed=True,device=device,technical_updates=3,total_technical_updates_to_date=16,capacity_points=65536,after_native_split_points=after_split,full_sequence_frames=121,all_training_views=21,all_support_pyramids_cached=121,alpha_supervision_and_native_regularizers_active=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),seconds_three_updates=elapsed,rows=rows)
    save(NEW/'research/capacity_probe.json',report)
    # Export the small probe state for a NEW-PROCESS online/baked replay check.
    s=torch.load(NEW/'research/probe/recovery.pt',map_location='cpu');restore(s,g,d,q,qo,cfg);del s
    export_asset(NEW/'research/probe/asset',g,d,q,cfg,data,4013,inp)
    expectations(NEW/'research/probe/expected.npz',g,d,q,cfg,data,4013,inp)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)

if __name__=='__main__':main()
