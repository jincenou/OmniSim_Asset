"""Equal extra-budget comparison of all-time and short-window fitting.

Same dense50k checkpoint, 15k schedule and frozen temporal parameters as the
single-time diagnostic. Only the actual training time support changes.
No complete dynamic-asset quality claim follows from selected-frame metrics.
"""
import argparse
from dataclasses import asdict, replace
import os
from pathlib import Path
import subprocess
import torch
from single_time_control import digest
from train_control import R, configuration, save
from simple_trainer_freetime_4d_pure_relocation import FreeTime4DRunner
from training_policy import TrainingPhase
from evaluate_asset import sha
from time_budget_evaluate import evaluate


class DiagnosticRunner(FreeTime4DRunner):
    def render_traj(self, step):
        save(Path(self.cfg.result_dir)/f'preview_scope_step{step}.json', {
            'state': 'selected_frame_evaluation_follows_training',
            'reason': 'Time-support diagnostic; no full dynamic asset is exported.'})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--label', choices=['walk063_rootlocal24','batch_0084_twentyfour'], required=True)
    parser.add_argument('--support', choices=['all','window21'], required=True)
    parser.add_argument('--check', action='store_true')
    args=parser.parse_args()
    frames=list(range(121)) if args.support=='all' else list(range(50,71))
    out=R/'diagnostics/time_budget'/f'{args.label}_{args.support}'
    source=R/'quality'/f'{args.label}_capacity50k/ckpts/ckpt_49999.pt'
    cfg,_,_=configuration(args.label,21,50000,out,recipe='capacity50k')
    cfg=replace(cfg,max_steps=15000,lr_schedule_steps=15000,ckpt_path=str(source),
        resume_mode='finetune',freeze_times=True,freeze_durations=True,
        phases=[TrainingPhase(start_step=0,freeze_times=True,freeze_durations=True,
            velocity_lr_factor=0.,allow_relocation=False)],
        use_relocation=False,relocation_stop_iter=0,save_steps=[5000,15000],
        eval_steps=[],export_ply=False,min_duration_frames=None,max_duration_frames=None)
    plan={'label':args.label,'support':args.support,'actual_training_frames':frames,
        'source_checkpoint':str(source),'source_sha256':sha(source),'steps':15000,
        'full_config':asdict(cfg),'scope':'Equal extra steps to single-time diagnostic; only training time support changes. Frozen dynamics are a diagnostic condition, not a recommended full-training recipe.'}
    if args.check:
        import json
        print(json.dumps(plan));return
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
    if gpu not in (4,5,6,7):raise ValueError('Unauthorized GPU')
    expected=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().removeprefix('GPU-')
    torch.cuda.init();actual=str(torch.cuda.get_device_properties(0).uuid)
    if actual!=expected or torch.cuda.device_count()!=1:raise RuntimeError('GPU UUID mismatch')
    out.mkdir(parents=True,exist_ok=False)
    save(out/'experiment.json',plan)
    save(out/'device.json',{'physical_gpu':gpu,'actual_uuid':actual,'pid':os.getpid()})
    runner=DiagnosticRunner(0,0,1,cfg)
    runner.trainset.pairs=[p for p in runner.trainset.pairs if p[1] in frames]
    runner.valset.pairs=[p for p in runner.valset.pairs if p[1] in frames]
    if len(runner.trainset)!=21*len(frames) or len(runner.valset)!=3*len(frames):
        raise ValueError('Wrong temporal split')
    loaded=torch.load(source,map_location='cpu',weights_only=False)
    if any(digest(v)!=digest(loaded['splats'][k]) for k,v in runner.splats.items()):
        raise ValueError('Initialization differs from frozen dense checkpoint')
    del loaded
    frozen={k:digest(runner.splats[k]) for k in ('times','durations','velocities')}
    save(out/'actual_training_pairs.json',{'pairs':runner.trainset.pairs,'frames':frames,
        'temporal_parameter_sha256':frozen,'original_normalized_timeline_preserved':True})
    runner.train()
    if runner.sampling_ledger.steps!=15000:raise ValueError('Incomplete optimization')
    if frozen!={k:digest(runner.splats[k]) for k in frozen}:raise ValueError('Frozen temporal parameters changed')
    save(out/'run_complete.json',{'state':'time_support_trained','steps':15000,'frames':frames,
        'frozen_parameters_unchanged':True,'sampling':runner.sampling_ledger.state()})
    runner.writer.close();del runner;torch.cuda.empty_cache()
    evaluation_frames=[30,60] if args.label=='walk063_rootlocal24' and args.support=='all' else [60]
    for frame in evaluation_frames:
        target=out/f'eval_f{frame:03d}';target.mkdir(exist_ok=False)
        evaluate([source,out/'ckpts/ckpt_14999.pt'],Path(cfg.data_dir),frame,target)


if __name__=='__main__':main()
