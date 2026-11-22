"""Freeze non-result-dependent inputs, cold initial anchors and paired exposure."""
import json,hashlib,random,shutil
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
NEW=Path(__file__).resolve().parents[1];ROOT=NEW.parent;BASE=ROOT.parent/'random8/r20261015_113332'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
    assert not (NEW/'PLAN.json').exists(), 'Never regenerate frozen experiment inputs'
    query=json.loads((ROOT/'query0_fused/PLAN.json').read_text());labels=['N04','N06','N11'];inputs={}
    for label in labels:
        out=NEW/'data'/label;out.mkdir(exist_ok=True);meta=BASE/'four_dg/data'/label/'fdanyone.json';d=json.loads(meta.read_text());init=meta.parent/d['canonical_npz'];cloud=np.load(init);points=cloud['points'].astype(np.float32);center=(points.min(0)+points.max(0))/2;length=float(np.linalg.norm(points.max(0)-points.min(0)));assert length>0
        chosen=[];distance=np.full(len(points),np.inf);index=int(np.square(points-points.mean(0)).sum(1).argmax())
        for k in range(128):
            chosen.append(index);distance=np.minimum(distance,np.square(points-points[index]).sum(1));index=int(distance.argmax())
        assert len(set(chosen))==128
        np.savez(out/'anchors.npz',anchors=points[chosen],indices=np.array(chosen),center=center,length=np.float32(length))
        train=d['train_views'];centers=np.array([c['camtoworld'] for c in d['cameras'] if c['view'] in train])[:,:3,3];direction=centers-center;direction/=np.linalg.norm(direction,axis=1,keepdims=True);support=[0]
        while len(support)<4:
            separation=(1-direction@direction[support].T).min(1);separation[support]=-1;support.append(int(separation.argmax()))
        support_views=[train[i] for i in support];assert set(support_views).isdisjoint([0,8,16])
        gen=random.Random(42);frames=[];stack=[]
        for step in range(1,33000):
            if not stack:stack=list(range(d['frame_count']))
            frame=stack.pop(gen.randrange(len(stack)));views=gen.sample(train,2);frames.append([frame,*views])
        save(out/'schedule.json',frames)
        config=json.loads((Path(query['anchors'][label]['fused']['run'])/'config.json').read_text());config.update(evq_label=label,evq_protocol='same-time two-target views;33k actual budget',old_assets_for_evaluation_only=True)
        save(out/'base_config.json',config);camera={k:d[k] for k in ['cameras','width','height','frame_count','time_denominator','train_views','heldout_views','source_to_normalized']};save(out/'cameras.json',camera)
        inputs[label]=dict(manifest=str(meta),manifest_sha256=sha(meta),initialization=str(init),initialization_sha256=sha(init),initial_points=len(points),anchor_file=str(out/'anchors.npz'),anchor_sha256=sha(out/'anchors.npz'),anchor_rule='Deterministic FPS from untouched pretraining cloud,first farthest from mean,ties first;no A/F point positions',L=length,center=center.tolist(),support_views=support_views,support_rule='Greedy farthest viewing directions from init bbox center,first lowest training view',frame_count=d['frame_count'],schedule_sha256=sha(out/'schedule.json'),base_config_sha256=sha(out/'base_config.json'),old_references=query['anchors'][label])
    plan=dict(schema='evq_v1_preflight_draft',state='implementation_and_ROI_audit_before_final_freeze',created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),task_sha256=sha(NEW/'research/task.txt'),labels=labels,arms=['R0','R1','R2','R3'],inputs=inputs,
      budget=dict(actual_updates=32999,shared_cold_warmup=2999,checkpoints=[2999,5000,15000,18000,32999],technical_probe_max_updates=200,technical_states_excluded=True,expected_terminals=12),
      model=dict(anchors=128,width=64,heads=2,support_views=4,scales=[2,4],channels=['R_gray_composite','G_gray_composite','B_gray_composite','soft_alpha','Sobel_luminance_x','Sobel_luminance_y'],tokens=32,cross_feature_offsets=[[-1,0],[1,0],[0,-1],[0,1]],learned_offset_max_feature_pixels=4,hard_mask='outside image or camera behind only',soft_prior='.05+.95*reference_alpha',null_prior='.05+1-mean(reference_alpha*hard_valid)',association='top4 learned canonical-attribute/content dot product plus squared distance prior sigma=.15L;recomputed after structure changes',decoder_hidden=[128,128],position_bound_fraction_L=.03,RGB_bound=.1,SH_C0=.28209479177387814,head_initialization='zero',enable_gate='clamp((actual_step-2999)/1000,0,1)',target_camera_independent=True),
      training=dict(targets_per_update=2,time_same=True,loss='mean two-view(.8 RGB L1+.2 DSSIM+fused_alpha_weight*balanced_alpha_L1)+one native regularizer+new-module regularizers',alpha_weight='.1*clamp((step-15000)/3000,0,1)',new_lr='5e-5+.5*(5e-4-5e-5)*(1+cos(pi*(step-3000)/29999))',new_lr_begin=5e-4,new_lr_end=5e-5,base_LR='same frozen fused complete absolute scheduler,initial LR then update after each step',new_regularizers=dict(position_normalized_square=1e-4,color_normalized_square=1e-4,offset_normalized_square=1e-4,provenance='Engineering constants proposed because referenced long parameter file absent; final frozen before quality training'),structure='native clone/split/prune/reset schedule and thresholds;two-view screen-gradient accumulation specified in frozen implementation'),
      evaluation=dict(H=[0,8,16],resolution=[352,640],background=.5,global_observations=39,old_cores='reuse fixed six-part ROI and flow denominators',expanded_ROI='reference conditioning lower-leg/foot and forearm/wrist capsules including background,requires fixed pretraining overlays',fixed_bad_windows={'N06':dict(view=16,frames=[22,46],part='left_leg'),'N11':dict(view=0,frames=[81,105],part='left_arm')},no_best_snapshot=True),
      gates=dict(provenance='Proposed engineering gates;explicit longer file not supplied;freeze before training',global_protection=dict(foreground_psnr_drop_max=.2,ssim_drop_max=.001,crop_lpips_increase_max=.002,each_core_coverage_drop_max=.01,each_temporal_rmse_ratio_max=1.05,outside_alpha_ratio_max=1.05),R3_vs_R0=dict(target_expanded_undercoverage_ratio_max=.9,target_expanded_RGB_RMSE_ratio_max=.95,target_expanded_positive_excess_ratio_max=1.05,local_LPIPS_increase_max=.002),R3_vs_R1_R2=dict(target_expanded_undercoverage_ratio_max=.95,target_expanded_RGB_RMSE_ratio_max=1.,target_expanded_positive_excess_ratio_max=1.05,local_LPIPS_increase_max=.002),unknown_not_pass=True,no_threshold_sweep=True,A='same old protections plus descriptive expanded metrics;not same-budget causal control'),
      assets=dict(table='float32 [T,128,64] for R1/R2/R3; R0 no latent branch',continuous_time='interpolate adjacent integer Z then shared decoder with base feature at actual t,online uses identical definition',replay_source_forbidden=str(BASE),tolerances=dict(max_abs_replay=1e-5,background_composition=1e-5),fractional_frames=[12.25,30.5],backgrounds=[0,.5,1]),
      resource=dict(authorized_physical_gpus=list(range(8)),share_other_tasks=True,never_terminate_others=True,minimum_disk_free_GiB=35,need_memory_probe=True,estimated_total_new_GiB=[50,60],user_offered_free_GiB=50),
      old_audits={k:dict(path=str(ROOT/k/'completion_audit.json'),sha256=sha(ROOT/k/'completion_audit.json')) for k in ['query0_fused','optimization_cross8_20261018','optimization_quality_v3_20261018']})
    save(NEW/'PLAN_DRAFT.json',plan);save(NEW/'STATUS.json',dict(state='implementation_preflight',training_started=False,formal_terminals=0,required_terminals=12,storage='user will free50GiB;recheck at admission',remaining=['implementation','frozen extended ROI and plan','mechanism and memory <=200-update tests','three cold warmups','12 full continuations','bake/replay/evaluate/media/audit']))
    print({l:dict(points=d['initial_points'],support=d['support_views'],frames=d['frame_count']) for l,d in inputs.items()},flush=True)
if __name__=='__main__':main()
