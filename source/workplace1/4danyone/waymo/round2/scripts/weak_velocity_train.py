"""Matched R07 15k runs: exactly one initialization parameter differs."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
import yaml

R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'backend/src'),str(R/'backend')]
from simple_trainer_freetime_4d_pure_relocation import Config,FreeTime4DRunner

def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['zero','body'],required=True);parser.add_argument('--sigma8',action='store_true');args=parser.parse_args()
    root=R/'diagnostics/weak_velocity';init_report=json.loads((root/'initialization.json').read_text())
    base=R/'reconstruction/outputs/r07_short61_24'
    out=root/(args.variant+('_sigma8' if args.sigma8 else ''))
    values=yaml.unsafe_load((base/'cfg.yml').read_text())
    init_path=Path(init_report['base_init' if args.variant=='zero' else 'new_init'])
    assert hashlib.sha256(init_path.read_bytes()).hexdigest()==init_report['base_sha256' if args.variant=='zero' else 'new_sha256']
    if args.sigma8:
        duration_record=json.loads((root/'duration_init.json').read_text())[args.variant]
        init_path=Path(duration_record['path'])
        assert hashlib.sha256(init_path.read_bytes()).hexdigest()==duration_record['sha256']
    values.update(result_dir=str(out),init_npz_path=str(init_path),max_steps=15000,lr_schedule_steps=15000,
        save_steps=[5000,15000],eval_steps=[15000],ckpt_path=None)
    cfg=Config(**values)
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);assert gpu in (4,5,6,7)
    expected=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().removeprefix('GPU-')
    torch.cuda.init();assert str(torch.cuda.get_device_properties(0).uuid)==expected and torch.cuda.device_count()==1
    out.mkdir(exist_ok=False)
    save(out/'experiment.json',{'variant':args.variant,'initial_sigma_frames':8 if args.sigma8 else 15,'config':asdict(cfg),'physical_gpu':gpu,'actual_cuda_uuid':expected,'pid':os.getpid(),
        'scope':'New15k schedule from initialization; no resume and no4DAnyone training. Paired comparison differs only initial velocities and output/input paths. Old5k uses a different schedule.'})
    runner=FreeTime4DRunner(0,0,1,cfg)
    assert len(runner.trainset)==21*61 and len(runner.valset)==3*61
    assert len(runner.splats['means'])==init_report['points']
    digests={key:hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for key,value in runner.splats.items()}
    save(out/'initial_parameters.json',{'parameter_sha256':digests,'points':init_report['points']})
    runner.train();assert runner.sampling_ledger.steps==15000
    save(out/'complete.json',{'state':'trained_exported_metrics_pending','steps':15000,'sampling':runner.sampling_ledger.state()})
    runner.writer.close()

if __name__=='__main__':main()
