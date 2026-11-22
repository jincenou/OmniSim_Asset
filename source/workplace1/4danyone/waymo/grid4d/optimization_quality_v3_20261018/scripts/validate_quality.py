"""Independent dynamic replay that rejects all reads from original Random8 data."""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
from quality_common import NEW,ROOT,load_asset,raw_render,camera,sha,save,verify,plan
BASE=ROOT.parent/'random8/r20261015_113332'


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--iteration',required=True,type=int);args=parser.parse_args();begin=time.monotonic();device=verify()
    def guard(event,values):
        if event=='open' and isinstance(values[0],(str,bytes)):
            assert not Path(os.fsdecode(values[0])).absolute().is_relative_to(BASE),'Standalone replay forbids Random8 reads'
    from pathlib import Path
    sys.addaudithook(guard);run=Path(args.run).resolve();assert run.is_relative_to(NEW)
    snapshot=json.loads((run/'snapshots'/f'{args.iteration}.json').read_text())
    for path,digest in snapshot['files'].items():assert sha(run/path)==digest
    model,deform,pipe,config=load_asset(run,args.iteration);data=json.loads((run/'asset_camera_data.json').read_text());refs=np.load(run/'snapshots'/f'{args.iteration}_reference.npz')
    errors=[];composition=[];mapped=[];captured=[]
    handle=deform.grid4d.register_forward_pre_hook(lambda module,a:captured.append(a[0][:1].detach().clone()))
    for key in refs.files:
        v,f=key.split('_');cam=camera(data,int(v[1:]),float(f[1:]));rgb,alpha=raw_render(model,deform,pipe,cam)
        error=float(np.abs(torch.cat([rgb,alpha]).cpu().numpy()-refs[key]).max());assert error<1e-5;errors.append(error)
        assert float(alpha.min())>=-1e-6 and float(alpha.max())<=1.000001
        for background in [.5,1.]:
            rendered,_=raw_render(model,deform,pipe,cam,background);delta=float((rendered-(rgb+background*(1-alpha))).abs().max());assert delta<1e-5;composition.append(delta)
        target=(float(cam.fid)*2*deform.grid4d.bound-deform.grid4d.bound)*.9;actual=float(captured[-1][0,3]);assert abs(target-actual)<1e-6
        mapped.append(dict(frame=float(f[1:]),normalized=float(cam.fid),internal=actual))
    handle.remove()
    paths=json.loads((run/'training_image_paths.json').read_text());assert paths and not any(f'/cam{v:02d}/' in path for path in paths for v in [0,8,16])
    save(run/'validation'/f'{args.iteration}.json',dict(state='passed',device=device,iteration=args.iteration,source_data_reads=0,source_data_guard='Every Python open under Random8 rejected',reference_count=len(errors),independent_replay_max_abs_error=max(errors),background_composition_max_abs_error=max(composition),time_checks=mapped,points=len(model.get_xyz),actual_optimizer_updates=snapshot['actual_optimizer_updates'],seconds=time.monotonic()-begin))
    print(run.name,args.iteration,'independent replay passed',max(errors),flush=True)


if __name__=='__main__':main()
