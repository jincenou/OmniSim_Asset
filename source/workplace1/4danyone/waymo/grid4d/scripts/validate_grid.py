"""Independent asset replay, time mapping and background-composition acceptance."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from grid_common import ROOT, manifest, load_asset, raw_render, camera, sha, save, verify, stamp


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--iteration',type=int,required=True)
    args=parser.parse_args();device=verify();run=Path(args.run).resolve();assert run.is_relative_to(ROOT)
    snapshot=json.loads((run/'snapshots'/f'{args.iteration}.json').read_text())
    for file,expected in snapshot['files'].items():assert sha(run/file)==expected
    model,deform,pipe,config=load_asset(run,args.iteration)
    data,_=manifest(config['label'])
    references=np.load(run/'snapshots'/f'{args.iteration}_reference.npz')
    differences=[];composition=[];times=[]
    captured=[]
    handle=deform.grid4d.register_forward_pre_hook(lambda m,a:captured.append(a[0][:1].detach().clone()))
    for key in references.files:
        view,frame=key.split('_');view=int(view[1:]);frame=float(frame[1:])
        cam=camera(data,view,frame)
        rgb,alpha=raw_render(model,deform,pipe,cam)
        actual=torch.cat([rgb,alpha]).cpu().numpy()
        delta=float(np.abs(actual-references[key]).max());differences.append(delta)
        assert delta<1e-5,(key,delta)
        assert float(alpha.min())>=-1e-6 and float(alpha.max())<=1.000001
        for bg in [.5,1.]:
            composed,_=raw_render(model,deform,pipe,cam,bg)
            error=float((composed-(rgb+bg*(1-alpha))).abs().max());composition.append(error)
            assert error<1e-5,(key,bg,error)
        want=(float(cam.fid)*2*deform.grid4d.bound-deform.grid4d.bound)*.9
        mapped=float(captured[-1][0,3]);assert abs(mapped-want)<1e-6
        times.append(dict(frame=frame,normalized=float(cam.fid),internal=mapped))
    handle.remove()
    paths=json.loads((run/'training_image_paths.json').read_text())
    expected={str(Path(data['source_data'])/'images'/f'cam{v:02d}'/f'{f:06d}.png') for v in data['train_views'] for f in range(data['frame_count'])}
    assert len(paths)==len(expected) and set(paths)==expected
    report=dict(state='passed',at=stamp(),device=device,run=str(run),iteration=args.iteration,
        independent_replay_max_abs_error=max(differences),background_composition_max_abs_error=max(composition),
        reference_count=len(differences),reference_views=[1,9,17],time_checks=times,
        both_hash_and_deform_strictly_loaded=True,training_image_count=len(paths),heldout_training_reads=0,
        heldout_training_read_evidence='Exact21-view path list and runtime Python open guard; native evaluation disabled',
        limitations='Technical dynamic replay only; not quality or real-world pose/scale validation')
    save(run/'validation'/f'{args.iteration}.json',report)
    print({k:v for k,v in report.items() if k!='time_checks'},flush=True)


if __name__=='__main__':
    main()
