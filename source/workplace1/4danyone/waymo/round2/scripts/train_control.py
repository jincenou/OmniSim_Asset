"""Controlled FTGS run using an isolated backend and a frozen baseline recipe."""
import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

R=Path(__file__).resolve().parents[1];W=R.parent
sys.path[:0]=[str(R/'backend/src'),str(R/'backend')]


def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


async def evaluate(output,checkpoint,data):
    command=[sys.executable,str(R/'scripts/evaluate_asset.py'),'--checkpoint',str(checkpoint),
             '--data',str(data),'--output',str(output/'quality_metrics')]
    with (output/'evaluation.log').open('xb') as log:
        process=await asyncio.create_subprocess_exec(*command,stdout=log,stderr=asyncio.subprocess.STDOUT)
        record={'pid':process.pid,'command':command,'state':'running'}
        save(output/'evaluation_process.json',record)
        code=await process.wait();(output/'evaluation.exit').write_text(str(code)+'\n')
        record.update(state='complete' if code==0 else 'failed',exit_code=code);save(output/'evaluation_process.json',record)
        if code:raise RuntimeError('Evaluation failed; trained checkpoint preserved, see evaluation.log')


def configuration(label,views,steps,output,resume=None,recipe='view_control'):
    import yaml
    from simple_trainer_freetime_4d_pure_relocation import Config
    # Local cfg.yml was emitted by our successful baseline runner, not supplied remotely.
    baseline=(R if label=='walk063_rootlocal24' else W)/'reconstruction/outputs'/label/'cfg.yml'
    values=yaml.unsafe_load(baseline.read_text())
    control=json.loads((R/'controls'/label/f'views{views}.json').read_text())
    init=Path(control['shared_init'])
    if hashlib.sha256(init.read_bytes()).hexdigest()!=control['shared_init_sha256']:
        raise RuntimeError('Control initialization changed')
    if recipe=='view_control' and label not in ('phone_full','pilot_043_twentyfour'):
        raise ValueError('Independent validation labels have no common-six view-count initialization')
    if recipe!='view_control':
        required_steps=50000 if recipe in ('budget_half_50k','relocation50k','capacity50k') else 15000
        if views!=21 or steps!=required_steps:raise ValueError(f'{recipe} requires21 views and{required_steps}-step schedule')
        data=Path(values['data_dir']);init=Path(values['init_npz_path'])
        if recipe=='budget_native':
            data=R/'quality_masters'/label/'native'
            manifest=json.loads((data/'manifest.json').read_text())
            if manifest['state']!='native_rgb_with_fixed_alpha_master_ready':raise ValueError('Native master is incomplete')
            if manifest['init_sha256']!=hashlib.sha256(init.read_bytes()).hexdigest():raise ValueError('Native control changes initialization')
        if recipe=='capacity50k':
            manifest=json.loads((R/'capacity'/label/'manifest.json').read_text())
            if manifest['state']!='dense_training_only_surface_ready':raise ValueError('Dense initialization is incomplete')
            if manifest['base_init_sha256']!=hashlib.sha256(init.read_bytes()).hexdigest():raise ValueError('Dense initialization has wrong Q0 source')
            init=Path(manifest['init'])
            if manifest['sha256']!=hashlib.sha256(init.read_bytes()).hexdigest():raise ValueError('Dense initialization changed')
        control={**control,'shared_data_dir':str(data),'shared_init':str(init),
            'shared_init_sha256':hashlib.sha256(init.read_bytes()).hexdigest(),
            'initialization_scope':'original Q0 initialization from21 training views; not the common-six view-count experiment',
            'dimensions':[704,1280] if recipe=='budget_native' else [352,640]}
        if recipe=='capacity50k':control['initialization_scope']='finer training21 visual-hull surface; same original normalization; capacity and surface sampling density change'
    control={**control,'optimization_steps':steps}
    values.update(data_dir=control['shared_data_dir'],init_npz_path=str(init),result_dir=str(output),
        train_view_ids=control['train_views'],validation_view_ids=control['holdout_views'],
        max_steps=steps,lr_schedule_steps=steps,save_steps=sorted(set([s for s in (5000,10000,15000,30000) if s<=steps]+[steps])),
        eval_steps=[steps],eval_sample_every=10,ckpt_path=resume,resume_mode='resume')
    # Relocation cutoff and all loss weights remain the baseline values.
    # Q1 changes the total budget and its explicit LR horizon; not a strict resume of Q0.
    if recipe=='relocation50k':values['relocation_stop_iter']=45000
    cfg=Config(**values)
    return cfg,baseline,control


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--label',choices=['phone_full','pilot_043_twentyfour','batch_0073_twentyfour','batch_0086_twentyfour','walk063_rootlocal24','batch_0084_twentyfour'],required=True)
    parser.add_argument('--views',type=int,choices=[6,12,21],required=True)
    parser.add_argument('--steps',type=int,choices=[2,4,5000,15000,50000],default=5000)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume');parser.add_argument('--check',action='store_true')
    parser.add_argument('--recipe',choices=['view_control','budget_half','budget_native','budget_half_50k','relocation50k','capacity50k'],default='view_control')
    args=parser.parse_args();cfg,baseline,control=configuration(args.label,args.views,args.steps,args.output,args.resume,args.recipe)
    if args.check:
        print(json.dumps({'config':asdict(cfg),'baseline':str(baseline),'control':control},default=str));return
    physical=int(os.environ['CUDA_VISIBLE_DEVICES'])
    if physical not in (4,5,6,7):raise RuntimeError('Unauthorized physical GPU')
    import torch
    from simple_trainer_freetime_4d_pure_relocation import FreeTime4DRunner
    expected=subprocess.check_output(['nvidia-smi',f'--id={physical}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().removeprefix('GPU-')
    torch.cuda.init();actual=str(torch.cuda.get_device_properties(0).uuid)
    if torch.cuda.device_count()!=1 or actual!=expected:raise RuntimeError('Actual CUDA device mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/'device.json',{'physical_gpu':physical,'actual_cuda_uuid':actual,'pid':os.getpid()})
    save(args.output/'experiment.json',{'label':args.label,'training_views':args.views,'steps':args.steps,
        'baseline_config':str(baseline),'baseline_sha256':hashlib.sha256(baseline.read_bytes()).hexdigest(),
        'control':control,'full_config':asdict(cfg),'recipe':args.recipe,
        'claim':('new run from common-six initialization' if args.recipe=='view_control' else 'new50k schedule with denser training-view visual-hull initialization and original PCA' if args.recipe=='capacity50k' else f'new{args.steps}-step schedule from unchanged Q0 initialization') if not args.resume else 'explicit checkpoint continuation',
        'relocation_cutoff':cfg.relocation_stop_iter,'generated_validation_not_real_ground_truth':True})
    runner=FreeTime4DRunner(0,0,1,cfg)
    if args.recipe=='capacity50k':
        expected_points=json.loads((R/'capacity'/args.label/'manifest.json').read_text())['points']
        if len(runner.splats['means'])!=expected_points or any(len(v)!=expected_points for v in runner.splats.values()):
            raise RuntimeError('Dense capacity was not loaded consistently into every dynamic parameter')
    digests={key:hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for key,value in runner.splats.items()}
    save(args.output/'initial_parameters.json',{'parameter_sha256':digests,'points':len(runner.splats['means']),
        'parser_transform':runner.parser.transform.tolist(),'state':'checkpoint_loaded' if args.resume else 'fresh_initialization'})
    runner.train()
    if runner.sampling_ledger.steps!=args.steps:raise RuntimeError('Training did not execute requested updates')
    save(args.output/'run_complete.json',{'state':'trained_exported_metrics_pending','steps':args.steps,
        'sampling':runner.sampling_ledger.state(),'visual_review':'pending'})
    if args.steps>=5000:
        # Release the training model before the independent evaluation process.
        runner.writer.close();del runner;torch.cuda.empty_cache()
        asyncio.run(evaluate(args.output,args.output/'ckpts'/f'ckpt_{args.steps-1}.pt',cfg.data_dir))
        if args.recipe!='view_control':
            async def intermediates():
                for step in (value-1 for value in cfg.save_steps if value<args.steps):
                    destination=args.output/f'evaluation_step{step}'
                    destination.mkdir(exist_ok=False)
                    await evaluate(destination,args.output/'ckpts'/f'ckpt_{step}.pt',cfg.data_dir)
            asyncio.run(intermediates())
        if args.views==6 and args.steps==5000:
            baseline_output=R/'quality'/f'baseline_{args.label}'
            baseline_output.mkdir(parents=True,exist_ok=False)
            save(baseline_output/'provenance.json',{'state':'frozen_existing_Q0_no_retraining',
                'source':str(baseline.parent),'note':'Uses its original initialization and PCA; distinct from the common-six view-count controls'})
            asyncio.run(evaluate(baseline_output,baseline.parent/'ckpts/ckpt_4999.pt',
                                 W/'reconstruction/data'/args.label))


if __name__=='__main__':main()
