"""Fixed-time fine-tuning diagnostic, never a complete dynamic asset.

Keep the dense50k state and normalized timeline; restrict actual loader pairs
to one source time. Freeze temporal parameters and velocity, disable relocation.
Only this selected time is evaluated or claimed after fine-tuning.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw
import torch
from train_control import R, configuration, save
from simple_trainer_freetime_4d_pure_relocation import FreeTime4DRunner
from training_policy import TrainingPhase
from evaluate_asset import Renderer, scalar_metrics, sha

class FixedTimeRunner(FreeTime4DRunner):
    def render_traj(self, step):
        # A full animated preview would imply validity outside the fitted time.
        save(Path(self.cfg.result_dir)/f'preview_scope_step{step}.json',
             {'state':'full_sequence_preview_not_applicable',
              'reason':'Single-time diagnostic; independent selected-frame evaluation follows training.'})

def digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

@torch.inference_mode()
def evaluate(paths, data, frame, out):
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    metric=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda().eval()
    models=[Renderer(p,data) for p in paths]
    if not np.array_equal(models[0].parser.transform,models[1].parser.transform):
        raise ValueError('Camera coordinate transforms changed')
    rows=[]
    for view in (0,8,16,1,9,17):
        source=data/'images'/f'cam{view:02d}'/f'{frame:06d}.png'
        rgba=np.asarray(Image.open(source).convert('RGBA'),dtype=np.float32)/255
        panels=[]
        for bg in (0.,.5,1.):
            target=rgba[...,:3]*rgba[...,3:]+bg*(1-rgba[...,3:])
            panels.append(target)
            for tag,model in zip(('dense50k','single_time'),models,strict=True):
                rgb,alpha=model.render(view,frame)
                pred=np.clip(rgb+bg*(1-alpha[...,None]),0,1)
                tensor=lambda x:torch.from_numpy(np.ascontiguousarray(x)).permute(2,0,1)[None].cuda()
                lp=float(metric(tensor(pred),tensor(target)));metric.reset()
                values=scalar_metrics(pred,target,alpha,rgba[...,3]>.5)
                if not all(np.isfinite(v) for v in (*values.values(),lp)):
                    raise ValueError('Nonfinite metric')
                rows.append(dict(view=view,frame=frame,background=bg,model=tag,lpips=lp,
                    split='validation_generated' if view in (0,8,16) else 'training_diagnostic',
                    reference_sha256=sha(source),**values))
                panels.append(pred)
        image=Image.fromarray(np.rint(np.concatenate(panels,axis=1)*255).astype(np.uint8))
        ImageDraw.Draw(image).text((5,5),f'v{view} frame{frame}; target / dense50k / fixed-time fit; black,gray,white',fill='red')
        image.save(out/f'view{view:02d}.jpg',quality=95)
    save(out/'metrics.json',{'state':'single_time_diagnostic_complete_visual_review_pending',
        'source_checkpoint':str(paths[0]),'source_sha256':sha(paths[0]),
        'checkpoint':str(paths[1]),'checkpoint_sha256':sha(paths[1]),'frame':frame,
        'samples':rows,'no_point_filtering':True,
        'scope':'Extra fixed-time fine-tuning from dense50k, not a from-scratch static baseline. Does not certify cross-time coherence or real Waymo accuracy.'})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--label',choices=['walk063_rootlocal24','batch_0084_twentyfour'],required=True)
    parser.add_argument('--frame',type=int,choices=[30,60],required=True)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    out=R/'diagnostics/single_time'/f'{args.label}_f{args.frame:03d}'
    source=R/'quality'/f'{args.label}_capacity50k/ckpts/ckpt_49999.pt'
    cfg,_,_=configuration(args.label,21,50000,out,recipe='capacity50k')
    cfg=replace(cfg,max_steps=15000,lr_schedule_steps=15000,ckpt_path=str(source),
        resume_mode='finetune',freeze_times=True,freeze_durations=True,
        phases=[TrainingPhase(start_step=0,freeze_times=True,freeze_durations=True,
            velocity_lr_factor=0.,allow_relocation=False)],
        use_relocation=False,relocation_stop_iter=0,save_steps=[5000,15000],
        eval_steps=[],export_ply=False,min_duration_frames=None,max_duration_frames=None)
    plan={'label':args.label,'source_checkpoint':str(source),'source_sha256':sha(source),
        'only_training_frame':args.frame,'steps':15000,'full_config':asdict(cfg),
        'scope':'Diagnostic fine-tuning of one time only, not dynamic asset generation. Frozen temporal parameters retain effective opacity at selected time. Extra budget is a confound; compare train and heldout separately.'}
    if args.check:
        print(json.dumps(plan));return
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
    if gpu not in (4,5,6,7):raise ValueError('Unauthorized GPU')
    expected=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().removeprefix('GPU-')
    torch.cuda.init();actual=str(torch.cuda.get_device_properties(0).uuid)
    if actual!=expected or torch.cuda.device_count()!=1:raise RuntimeError('GPU UUID mismatch')
    out.mkdir(parents=True,exist_ok=False)
    save(out/'experiment.json',plan);save(out/'device.json',{'physical_gpu':gpu,'actual_uuid':actual,'pid':os.getpid()})
    runner=FixedTimeRunner(0,0,1,cfg)
    runner.trainset.pairs=[pair for pair in runner.trainset.pairs if pair[1]==args.frame]
    runner.valset.pairs=[pair for pair in runner.valset.pairs if pair[1]==args.frame]
    if len(runner.trainset)!=21 or len(runner.valset)!=3:raise ValueError('Wrong fixed-time split')
    loaded=torch.load(source,map_location='cpu',weights_only=False)
    if any(digest(v)!=digest(loaded['splats'][k]) for k,v in runner.splats.items()):
        raise ValueError('Initialization differs from frozen dense checkpoint')
    del loaded
    frozen={k:digest(runner.splats[k]) for k in ('times','durations','velocities')}
    save(out/'actual_training_pairs.json',{'pairs':runner.trainset.pairs,'frame':args.frame,'time':args.frame/120,'temporal_parameter_sha256':frozen})
    runner.train()
    if runner.sampling_ledger.steps!=15000:raise ValueError('Incomplete optimization')
    if frozen!={k:digest(runner.splats[k]) for k in frozen}:raise ValueError('Frozen temporal parameters changed')
    save(out/'run_complete.json',{'state':'fixed_time_trained','steps':15000,'frame':args.frame,
        'frozen_parameters_unchanged':True,'sampling':runner.sampling_ledger.state()})
    runner.writer.close();del runner;torch.cuda.empty_cache()
    evaluate([source,out/'ckpts/ckpt_14999.pt'],Path(cfg.data_dir),args.frame,out)

if __name__=='__main__':main()
