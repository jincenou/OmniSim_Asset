"""Optimize the local walking asset with measured view/time sampling."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
import yaml

ROUND=Path(__file__).resolve().parents[1];W=ROUND.parent;R=ROUND/'reconstruction'
sys.path[:0]=[str(ROUND/'backend/src'),str(ROUND/'backend')]
from simple_trainer_freetime_4d_pure_relocation import Config,FreeTime4DRunner


def main():
    plan=Path(os.environ.get('FDANYONE_ASSET_PLAN',str(ROUND/'short24_plan.json')))
    label=os.environ['FDANYONE_ASSET_NAME'];assert label in {'moving_R01_short61_M0_24','moving_R01_short61_M2_24'} | {x['asset_label'] for x in json.loads(plan.read_text())['jobs']};out=R/'outputs'/label;data=R/'data'/label
    physical=int(os.environ['CUDA_VISIBLE_DEVICES']);assert physical in (4,5,6,7)
    expected=subprocess.check_output(['nvidia-smi',f'--id={physical}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().removeprefix('GPU-')
    torch.cuda.init();assert str(torch.cuda.get_device_properties(0).uuid)==expected
    values=yaml.unsafe_load((W/'reconstruction/outputs/phone_full/cfg.yml').read_text())
    values.update(data_dir=str(data),init_npz_path=str(data/'init.npz'),result_dir=str(out),
        train_view_ids=[i for i in range(24) if i not in (0,8,16)],validation_view_ids=[0,8,16],
        end_frame=61,render_traj_n_frames=61,render_traj_time_frames=61,max_steps=5000,lr_schedule_steps=5000,save_steps=[1000,3000,5000],eval_steps=[5000],eval_sample_every=10,ckpt_path=None)
    import numpy as np
    init=np.load(data/'init.npz');assert int(init['frame_end'])==61 and int(init['time_denominator'])==60
    assert np.isfinite(init['durations']).all() and (init['durations']>0).all()
    cfg=Config(**values);out.mkdir(parents=True,exist_ok=False)
    (out/'experiment.json').write_text(json.dumps({'scope':'root-local walking dynamic asset from24 generated views',
        'full_config':asdict(cfg),'init_sha256':hashlib.sha256((data/'init.npz').read_bytes()).hexdigest(),
        'physical_gpu':physical,'actual_cuda_uuid':str(torch.cuda.get_device_properties(0).uuid),'pid':os.getpid(),
        'motion_initialization':'five visual hull times (0,15,30,45,60); initial sigma0.25 normalized =0.6s; zero initial velocity learned during optimization; not material-point correspondences'},indent=2)+'\n')
    runner=FreeTime4DRunner(0,0,1,cfg)
    assert len(runner.trainset)==21*61 and len(runner.valset)==3*61
    assert {p[1] for p in runner.trainset.pairs}==set(range(61))
    runner.train()
    assert runner.sampling_ledger.steps==5000
    (out/'training_complete.json').write_text(json.dumps({'steps':5000,'sampling':runner.sampling_ledger.state()},indent=2)+'\n')


if __name__=='__main__':main()
