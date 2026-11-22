"""One-parameter controls against frozen dense50k walking baselines."""
import argparse
import asyncio
from dataclasses import asdict,replace,is_dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import torch
import yaml
from train_control import configuration,evaluate,save,R
from simple_trainer_freetime_4d_pure_relocation import FreeTime4DRunner

def main():
    p=argparse.ArgumentParser();p.add_argument('--label',choices=['walk063_rootlocal24','batch_0084_twentyfour'],required=True)
    p.add_argument('--variant',choices=['relocate45k','duration8'],required=True);p.add_argument('--check',action='store_true');a=p.parse_args()
    out=R/'quality'/f'{a.label}_dense_{a.variant}'
    cfg,baseline,control=configuration(a.label,21,50000,out,recipe='capacity50k')
    before=asdict(cfg)
    cfg=replace(cfg,**({'relocation_stop_iter':45000} if a.variant=='relocate45k' else {'max_duration_frames':8.0}))
    after=asdict(cfg);changes={k:{'before':before[k],'after':v} for k,v in after.items() if before[k]!=v}
    assert set(changes)==({'relocation_stop_iter'} if a.variant=='relocate45k' else {'max_duration_frames'})
    frozen=R/'quality'/f'{a.label}_capacity50k'
    recorded=yaml.unsafe_load((frozen/'cfg.yml').read_text())
    recorded={k:asdict(v) if is_dataclass(v) else v for k,v in recorded.items()}
    difference={k:(recorded.get(k),v) for k,v in before.items() if recorded.get(k)!=v}
    assert set(difference)<= {'result_dir'},difference
    plan={'baseline':str(frozen),'label':a.label,'variant':a.variant,'changes':changes,
        'same':['dense initialization file','generated RGBA observations','21/3 view split','50000-step learning-rate schedule','seed','loss weights','point budget'],
        'scope':'New full optimization, not resume. No4DAnyone retraining; generated observations are not true novel-view GT.',
        'duration8_caveat':'8/25=0.32s maximum Gaussian temporal sigma. Bound applies during initialization and optimization, so initial duration parameters change; no positions/velocities replaced.' if a.variant=='duration8' else None,
        'full_config':after,'control':control}
    if a.check:print(json.dumps(plan));return
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);assert gpu in (4,5,6,7)
    expected=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().removeprefix('GPU-')
    torch.cuda.init();actual=str(torch.cuda.get_device_properties(0).uuid)
    assert torch.cuda.device_count()==1 and actual==expected
    out.mkdir(exist_ok=False);save(out/'experiment.json',plan);save(out/'device.json',{'physical_gpu':gpu,'actual_cuda_uuid':actual,'pid':os.getpid()})
    runner=FreeTime4DRunner(0,0,1,cfg)
    expected_points=json.loads((R/'capacity'/a.label/'manifest.json').read_text())['points']
    assert all(len(v)==expected_points for v in runner.splats.values())
    digests={k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in runner.splats.items()}
    original=json.loads((frozen/'initial_parameters.json').read_text())['parameter_sha256']
    changed=[k for k,v in digests.items() if original[k]!=v]
    assert set(changed)<=({'durations'} if a.variant=='duration8' else set()),changed
    save(out/'initial_parameters.json',{'parameter_sha256':digests,'changed_from_dense50k':changed,'points':expected_points})
    runner.train();assert runner.sampling_ledger.steps==50000
    save(out/'run_complete.json',{'state':'trained_exported_metrics_pending','steps':50000,'sampling':runner.sampling_ledger.state()})
    runner.writer.close();del runner;torch.cuda.empty_cache()
    asyncio.run(evaluate(out,out/'ckpts/ckpt_49999.pt',cfg.data_dir))

if __name__=='__main__':main()
