"""Freeze exact starts, paired samples, absolute LR curves and all decision rules."""
import csv
import hashlib
import json
import math
import random
import re
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch

NEW=Path(__file__).resolve().parents[1];ROOT=NEW.parent


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    assert not (NEW/'PLAN.json').exists(),'Frozen plan already exists: inspect and resume, do not replace'
    dependencies=['optimization/FINDINGS.md','optimization/PLAN.json','optimization/STATUS.json','optimization/EVALUATION_STATUS.json','optimization/alpha_gradient_acceptance.json','optimization/comparison/index.html','audit/inventory.json','audit/paired_summary.csv','audit/budget_quality.csv','audit/costs.csv','mechanism/static/results/summary.json','ASSETS.json','completion_audit.json','README_run.md','scripts/optimize_grid.py','scripts/grid_common.py','scripts/evaluate_grid.py','scripts/part_metrics.py','scripts/prepare_diagnostics.py','scripts/replay_asset.py','source/scene/gaussian_model.py','source/scene/deform_model.py','source/gaussian_renderer/__init__.py','source/utils/general_utils.py']
    save(NEW/'research/dependencies.json',{p:dict(path=str(ROOT/p),sha256=sha(ROOT/p)) for p in dependencies})
    old=json.loads((ROOT/'optimization/PLAN.json').read_text());coefficient=float(re.search(r'Add([0-9.]+)',old['loss']['alpha'])[1]);assert coefficient==.1
    identities={};lr_rows=[];anchors=[]
    for label,points in [('N04',30205),('N11',14134)]:
        run=ROOT/'optimization/runs'/f'{label}_alpha';config=json.loads((run/'config.json').read_text());state=torch.load(run/'recovery.pt',map_location='cpu',weights_only=False)
        complete=json.loads((run/'complete.json').read_text());snapshot=json.loads((run/'snapshots/48000.json').read_text())
        assert complete['total_actual_updates']==snapshot['actual_optimizer_updates']==47999 and state['extra_updates']==15000
        assert len(state['gaussians']['_xyz'])==points
        data=json.loads((run/'asset_camera_data.json').read_text());frames=data['frame_count'];views=data['train_views'];assert len(views)==21 and not set(views)&{0,8,16}
        manifest_path=Path(config['provenance']['manifest']);assert sha(manifest_path)==config['provenance']['manifest_sha256']
        manifest=json.loads(manifest_path.read_text());assert manifest['frame_count']==frames and manifest['train_views']==views
        data['source_data']=manifest['source_data']
        uids=[v*frames+f for v in views for f in range(frames)]
        directory=NEW/'data'/label;directory.mkdir(parents=True,exist_ok=True)
        train=[dict(view=v,frame=f,path=str(Path(data['source_data'])/'images'/f'cam{v:02d}'/f'{f:06d}.png')) for v in views for f in range(frames)]
        save(directory/'training_images.json',train)
        samples=np.rint(np.linspace(0,frames-1,13)).astype(int).tolist()
        evaluation=[dict(view=v,frame=f,path=str(Path(data['source_data'])/'images'/f'cam{v:02d}'/f'{f:06d}.png')) for v in [0,8,16] for f in samples]
        save(directory/'evaluation_images.json',evaluation)
        evidence={p:dict(path=str(run/p),sha256=sha(run/p)) for p in ['recovery.pt','config.json','complete.json','asset_camera_data.json','snapshots/48000.json','snapshots/48000_reference.npz','validation/48000.json']}
        for p,digest in snapshot['files'].items():assert sha(run/p)==digest;evidence[p]=dict(path=str(run/p),sha256=digest)
        mapping={}
        for optimizer in ['gaussian_optimizer','deform_optimizer']:
            o=state[optimizer];groups=[]
            for g in o['param_groups']:
                records=[]
                for pid in g['params']:
                    s=o['state'][pid];assert set(['step','exp_avg','exp_avg_sq'])<=s.keys()
                    assert torch.isfinite(s['exp_avg']).all() and torch.isfinite(s['exp_avg_sq']).all()
                    records.append(dict(id=pid,shape=list(s['exp_avg'].shape),step=float(s['step']),exp_avg_sha256=hashlib.sha256(s['exp_avg'].numpy().tobytes()).hexdigest(),exp_avg_sq_sha256=hashlib.sha256(s['exp_avg_sq'].numpy().tobytes()).hexdigest()))
                groups.append(dict(name=g['name'],lr=g['lr'],eps=g['eps'],betas=list(g['betas']),state=records))
            mapping[optimizer]=groups
        repeat_seed=20261018 if config['seed']!=20261018 else 20261019
        sequence={}
        for trial in ['primary','repeat']:
            rng=random.Random();rng.setstate(state['rng']['python']) if trial=='primary' else rng.seed(repeat_seed)
            stack=list(state['stack']) if trial=='primary' else list(uids)
            start_rng=rng.getstate();start_stack=list(stack);rows=[];checkpoints={}
            for j in range(5000):
                if not stack:stack=list(uids)
                uid=stack.pop(rng.randrange(len(stack)));view,frame=divmod(uid,frames)
                rows.append(dict(stage_update=j+1,global_actual_updates=48000+j,uid=uid,view=view,frame=frame))
                if j+1 in [1000,3000,5000]:checkpoints[j+1]=dict(python_rng=rng.getstate(),stack=list(stack))
            p=directory/f'{trial}_samples.json';save(p,rows)
            checkpoint_path=directory/f'{trial}_sampler_states.pt';torch.save(dict(start_rng=start_rng,start_stack=start_stack,checkpoints=checkpoints),checkpoint_path)
            sequence[trial]=dict(path=str(p),sha256=sha(p),state_path=str(checkpoint_path),state_sha256=sha(checkpoint_path),python_seed='restored_alpha_final_rng' if trial=='primary' else repeat_seed)
        assert sequence['primary']['sha256']!=sequence['repeat']['sha256']
        optim=config['optimization'];groups=[g for name in ['gaussian_optimizer','deform_optimizer'] for g in state[name]['param_groups']]
        curves=[]
        for j in range(5000):
            k=math.floor(14999*j/4999);virtual=15000+k*17999/14999
            values={g['name']:.25*g['lr'] for g in groups}
            for name,scale,horizon,finalscale in [('xyz',5,optim['position_lr_max_steps'],5),('deform',optim['network_lr_scale'],optim['deform_lr_max_steps'],1),('grid',optim['grid_lr_scale'],optim['deform_lr_max_steps'],optim['grid_lr_scale'])]:
                init=optim['position_lr_init']*scale;final=optim['position_lr_final']*finalscale;t=np.clip(virtual/horizon,0,1)
                values[name]=.25*float(np.exp(np.log(init)*(1-t)+np.log(final)*t)) if init and final else 0.
            curves.append(dict(stage_update=j+1,old_stage_index=k,old_virtual_iteration=virtual,**values))
        p=directory/'learning_rates.json';save(p,curves)
        identities[label]=dict(start_run=str(run),global_actual_updates=47999,points=points,frames=frames,source_seed=config['seed'],repeat_seed=repeat_seed,files=evidence,optimizer_groups=mapping,
            counters_note='Global optimizer-call counter differs from Gaussian Adam step47963 and deformation45000; preserve actual per-parameter moments/counters. Historical structural replacement can remove gradients on a call; do not overwrite Adam counters with global label.',
            data=dict(training=str(directory/'training_images.json'),training_sha256=sha(directory/'training_images.json'),evaluation=str(directory/'evaluation_images.json'),evaluation_sha256=sha(directory/'evaluation_images.json'),frozen_core_cache=str(ROOT/'optimization/data'/label)),
            sampling=sequence,lr=dict(path=str(p),sha256=sha(p),first=curves[0],last=curves[-1]))
        targets=[(1,0,'head'),(9,(frames-1)//2,'head'),(17,frames-1,'head'),(1,(frames-1)//2,'torso'),(9,0,'torso'),(17,frames-1,'torso')] if label=='N04' else [(9,0,'left_arm'),(9,3*(frames-1)//4,'left_arm'),(17,(frames-1)//4,'right_arm'),(17,frames-1,'right_arm'),(1,(frames-1)//2,'torso'),(17,0,'torso')]
        for i,(view,frame,part) in enumerate(targets):
            times=sorted({frame+d for d in [-1,1,-5,5,-15,15,-30,30] if 0<=frame+d<frames},key=lambda f:(abs(f-frame),f))[:6]
            donors=[dict(view=view,frame=t,type='same_view_time') for t in times]
            near=sorted([v for v in views if v!=view],key=lambda v:(min((v-view)%24,(view-v)%24),v))
            donors.extend(dict(view=v,frame=frame,type='same_time_view') for v in near[:8-len(donors)])
            anchors.append(dict(id=f'{label}_{i:02d}',label=label,view=view,frame=frame,part=part,donors=donors,
                selection='Deterministic train-only view/time/body-part anchors, before boundary results; normal/difficult classification from measured starting residual, not heldout scores'))
    save(NEW/'data/anchors.json',anchors)
    plan=dict(schema=1,created_at=datetime.now(timezone.utc).isoformat(),request_sha256=sha(NEW/'REQUEST.txt'),source_commit='3ab7b74482910d22fba7d51ac60c8cae1fe24bf1',identities=identities,
        arms=dict(A='L0 =0.8 global RGB L1+0.2 DSSIM+0.1 global balanced alpha L1+old spatial/temporal feature regularizers',U='L0+delta*uniform alpha L1',B='L0+delta*band alpha L1'),lambda_alpha=coefficient,delta=coefficient,
        loss=dict(global_old='.5 mean(abs(pred_alpha-soft_ref_alpha) over ref>.5)+.5 mean over ref<=.5',uniform='.5 mean(e[M])+.5 mean(e[~M])',band='.5 mean(e[B_in])+.5 mean(e[B_out])',new_weight_mass=1.,empty_policy='U and B both skip ONLY added term if either boundary side is empty; L0 retained',rgb='unchanged global RGB; no parts',alpha='one all-one-color black-background pass shared by all alpha terms; exactly2 training renders/update',regularizers='original spatial and temporal .5 each; exact original paths'),
        regions=dict(threshold=.5,comparison='>=',radius_pixels=3,kernel='Euclidean integer disk dx^2+dy^2<=9',padding='explicit zero outside image for both erosion/dilation',interior='erode(M,3)',inner='M minus erode(M,3)',outer='dilate(M,3) minus M',far='complement of dilate(M,3)',soft_target=True,training_size=[352,640],immutable=True),
        schedule=dict(stage_updates=5000,start_global_actual=47999,end_global_actual=52999,checkpoints=[1000,3000,5000],checkpoint_labels=[49000,51000,53000],snapshot_timing='after actual optimizer update',main_comparison='fixed last snapshot; no best selection',lr='floor(14999*j/4999) of old15000-stage curve, multiplied0.25 for EVERY group; frozen absolute curves referenced above',structure=dict(densify=False,prune=False,opacity_reset=False),parameter_groups='All previous trainable Gaussian,SH,hash,deform groups unchanged; Adam state kept'),
        evaluation=dict(background=.5,resolution=[352,640],heldout=[0,8,16],sampled_times='same old13, exact crop/hash denominators',reference='Frozen generated development set; not real Waymo novel-view truth',primary_metrics=['full_psnr','foreground_psnr','ssim','crop_lpips','silhouette_iou','outside_alpha_mean'],regions='above same rule; full-frame rendered output remains unmasked',parts='existing fixed conservative six-core ROIs; N11 arms separate; head core not face accuracy',flow='reuse old gated estimated reference flow, report valid coverage',unknown='NA; never convert missing reference/correspondence into zero error'),
        gates=dict(basic_comparators=['A','common_alpha_start'],outside_alpha_relative_max=.90,coverage_drop_max=.01,crop_lpips_increase_max=.002,foreground_psnr_drop_max=.2,silhouette_iou_drop_max=.003,ssim_drop_max=.001,boundary_independent_comparator='U',boundary_independent_same_limits=True,require_each_identity=True,require_all_reliable_parts=True,visual='No new obvious color contamination,body erosion or motion anomaly; far background excess must not increase beyond numerical1e-6',engineering_not_significance=True),
        repeat=dict(trigger='Same U passes basic gates on both identities OR same B passes basic+independent gates on both; visual gate required',max_rounds=1,arms=['A','U','B'],restore='original alpha47999 each identity, not primary52999',reset='Only common RNG and initial sampling stack; Gaussian,hash,deform,Adam unchanged',seed='per-identity repeat_seed above',failure_no_seed_retry=True),
        technical=dict(replay_tolerance=1e-5,gradient_relative_tolerance=.01,zero_gradient='abs analytic,numeric <=1e-7 is zero/NA, not positive-gradient pass; otherwise compare relative at stable contributions; preserve discontinuities',low_opacity_threshold=.01,fixed_support='reference-core projected center assignment plus raster visibility; conditional support, not material correspondence'),
        probes=dict(anchors_path=str(NEW/'data/anchors.json'),anchors_sha256=sha(NEW/'data/anchors.json'),max_anchors_per_identity=6,appearance='Independent SH-only autograd, no optimizer.step; hold view or time fixed; common visible contributing Gaussian subset',donor_max=8,query_training=False,target_color_ranking=False,query_mapping='Conditional canonical-Gaussian/depth correspondence with cycle/occlusion/foreground gates; target colors only scored after ranking',query_thresholds=dict(min_alpha=.5,cycle_pixels=2.,relative_depth=.03,absolute_depth=.01,min_transfer_pixels=64),query_rank='Visibility/cycle/depth support and donor-only sharpness, never target RGB error',flow_scope='Reference optical-flow validity reused for evaluation; no heldout donor access'),
        runtime=dict(gpu_pool=list(range(8)),concurrency='bounded asyncio subprocesses,one project training per card plus permitted existing jobs',disk_reserve_GiB=35,train_free_memory_MiB=16000,diagnostic_free_memory_MiB=6000,tmux_server='fdgrid_boundary',dependencies=['CPU prepare and frozen plan','exact restore/replay and new loss gradient acceptance','primary6 training','independent3 checkpoints each','final metrics+full6-column media+gate','conditional repeat6','interpretation and delivery']),
        not_executed_this_round=['New identities N01/N07','New33k prefixes','LearnedQuery','Front-end changes','Capacity growth','Texture loss','Automatic mask cleanup'],maximum_new_training_runs=12)
    save(NEW/'PLAN.json',plan);(NEW/'plan_hash.txt').write_text(sha(NEW/'PLAN.json')+'\n')
    save(NEW/'STATUS.json',dict(state='plan_frozen_preparation_pending',at=plan['created_at'],jobs={},plan_sha256=sha(NEW/'PLAN.json')))
    print('frozen plan',sha(NEW/'PLAN.json'),'both full alpha recoveries present;12 train anchors;paired sequences and LR curves saved',flush=True)


if __name__=='__main__':main()
