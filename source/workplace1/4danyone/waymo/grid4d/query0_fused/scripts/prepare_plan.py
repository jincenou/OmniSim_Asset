import datetime, hashlib, json, shutil
from pathlib import Path

NEW=Path(__file__).resolve().parents[1]; ROOT=NEW.parent
CROSS=ROOT/'optimization_cross8_20261018'; V3=ROOT/'optimization_quality_v3_20261018'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()
def save(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
    assert not (NEW/'PLAN.json').exists()
    for folder in ['runs','results','logs','research','cache/raw','data','demand_tables','sampling_traces','comparison']:
        (NEW/folder).mkdir(parents=True,exist_ok=True)
    labels=['N01','N02','N03','N04','N06','N07','N08','N11']; anchors={}
    prior=json.loads((CROSS/'ASSETS.json').read_text())['assets']
    for label in labels:
        rec={}
        for method in ['A','fused']:
            a=next(a for a in prior if a['label']==label and a['method']==method)
            it=53000 if method=='A' else 33000;run=Path(a['asset']);result=Path(a['results'])
            raw=result.parent.parent/'cache/raw'/result.name
            if label in ['N04','N11'] and method=='A':raw=ROOT/'optimization_boundary_20261018/cache/raw/primary'/f'{label}_A'
            rec[method]=dict(run=str(run),iteration=it,metrics=str(result/'metrics.json'),parts=str(result/'parts.csv'),raw=str(raw),snapshot_sha256=sha(run/'snapshots'/f'{it}.json'))
        if label in ['N01','N07']:
            run=V3/'runs'/f'{label}_alpha';result=V3/'results'/f'{label}_alpha_48000';raw=V3/'cache/raw'/result.name;parts=result/'parts.csv'
        elif label in ['N04','N11']:
            run=ROOT/'optimization/runs'/f'{label}_alpha';result=ROOT/'optimization/results'/run.name;raw=ROOT/'optimization/raw'/run.name;parts=result/'grid4d_alpha.csv'
        else:
            run=CROSS/'runs'/f'{label}_alpha';result=CROSS/'results'/f'{label}_alpha_48000';raw=NEW/'cache/raw'/result.name;parts=NEW/'results'/result.name/'parts.csv'
        rec['alpha']=dict(run=str(run),iteration=48000,metrics=str(result/'metrics.json'),parts=str(parts),raw=str(raw),snapshot_sha256=sha(run/'snapshots/48000.json'),needs_dense=label in ['N02','N03','N06','N08'])
        manifest=ROOT.parent/'random8/r20261015_113332/four_dg/data'/label/'fdanyone.json';m=json.loads(manifest.read_text())
        assert m['train_views']==[v for v in range(24) if v not in [0,8,16]]
        rec['input']=dict(manifest=str(manifest),sha256=sha(manifest),frame_count=m['frame_count'],train_views=m['train_views'],width=m['width'],height=m['height'],time_denominator=m['time_denominator'])
        anchors[label]=rec
    for label in ['N04','N06','N11']:
        for method in ['fused','A']:
            d=anchors[label][method];run=Path(d['run']);snapshot=json.loads((run/'snapshots'/f"{d['iteration']}.json").read_text())
            for path,h in snapshot['files'].items():assert sha(run/path)==h,(label,method,path)
            d.update(recovery_sha256=sha(run/'recovery.pt'),points=snapshot['points'])
    request=Path('/home/ojc/.codex/attachments/8a362598-e93e-492b-80db-0a8d12e684d9/pasted-text-1.txt')
    shutil.copyfile(request,NEW/'research/task.txt')
    lr=ROOT/'optimization_boundary_20261018/data/N04/learning_rates.json';schedule=json.loads(lr.read_text())[:2000];assert len(schedule)==2000
    save(NEW/'data/learning_rates.json',schedule)
    for name in ['quality_common.py','evaluate_quality.py','evaluate_parts.py','validate_quality.py','publish_quality.py','profile_replay.py']:
        shutil.copyfile(CROSS/'scripts'/name,NEW/'scripts'/name)
    plan=dict(schema='query0_fused_v1',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),request_sha256=sha(request),labels=labels,query_labels=['N04','N06','N11'],reserve=['N11'],failed_frontend_preserved=['N05','N09','N10'],anchors=anchors,
        roles=dict(research_base='fused32999',quality_reference='A52999',historical='HUSTgrid61'),
        budget=dict(extra_updates=2000,snapshots=[500,1000,2000],fused_final=34999,A_final=54999,initial_branches=12,conditional_A_branches=4),
        selection=dict(parts=['head','torso','left_arm','right_arm','left_leg','right_leg'],bins='numpy.array_split(arange(T),8); lower middle actual slot of each contiguous bin',minimum_core_pixels=64,
            score_pool='21 train views x8 representative times; initial shared once and branch refresh after1000',demand='.5*median(u)+.5*median(e/(e+part_median_e+1e-6)) per valid(part,bin)',undercoverage='sum(max(ref_alpha-pred_alpha,0))/max(sum(ref_alpha),1e-6)',
            E='Same normalized foreground RGB L1 / undercoverage; nearest same-view bin score is proxy for nonscored frames',
            U='Restore native Python RNG and stack; pregenerate2000 base items',P='Uniform valid(part,bin), then shared visibility/exposure weighted native observation',Q='demand(part,bin)+1e-3, then same visibility/exposure weighted native observation',
            secondary='min(area/(median_valid_area_in_cell+1e-6),2)/sqrt(1+branch_exposure)',even_steps='1-based2,4,... use same base sequence index as U; odd steps adaptive',unknown='NA is not zero; empty priority pool uses paired uniform item and logs reason',selector_rng='Private numpyPCG64 seed42; Torch perturbation RNG restored and isolated from selection/diagnostics',importance_correction=False),
        loss='Unchanged .8RGB L1+.2DSSIM+.1 balanced softalpha L1+original spatial/temporal feature regularization',
        optimization=dict(full_restore=True,fixed_point_capacity=True,all_original_parameters_trainable=True,no_densification_pruning_or_reset=True,active_SH_degree_unchanged=True,absolute_LR_source=str(lr),absolute_LR_source_sha256=sha(lr),schedule_sha256=sha(NEW/'data/learning_rates.json'),schedule='First2000 exact rows; no remapping or amplitude change'),
        evaluation=dict(H=[0,8,16],H_excluded_from_selection=True,reference='Frozen generated development reference; not realWaymoGT or independenttest',global_protocol='3views x13 fixed times,gray.5,352x640,original crops/AlexNetLPIPS',parts='All6 fixed reference cores and full timeline/unchanged reference flow gates',curve_not_selected_on_H=True,worst_window='For each view/part all consecutive25-slot windows: reference-core-pixel-weighted missing coverage andRGBMSE,valid slots/NA recorded'),
        gates=dict(foreground_psnr_drop=.2,ssim_drop=.001,lpips_increase=.002,coverage_drop=.01,temporal_rmse_ratio=1.05,outside_alpha_ratio=1.05,target_parts={'N06':['left_leg','right_leg'],'N11':['left_arm','right_arm']},
            Q_vs_U='Per targetidentity:side with lower U mean coverage (left on tie) improves>=.02; other side drop<=.01; all3identities all protectionlines; scoped visual review',
            Q_vs_P_E='Per targetidentity:min(Qleft,Qright)-min(comparatorleft,comparatorright)>=.005 for BOTH P/E, same complete protectionlines',
            A_confirmation='Automatically launch N06/N11 A+U/Q only after all initial numeric conditions pass; visual unknown preserved',alpha48='All old protectionlines versusA; no individual checkpoint selection',unknown_is_not_pass=True),
        runtime=dict(environment='/mnt/ojc/miniconda3/envs/grid4d',gpus=list(range(8)),tmux='fdgrid_query0',one_owned_job_per_gpu=True,sharing_allowed=True,minimum_free_MiB=16000,disk_reserve_GiB=35,phase_write_headroom_GiB=5),
        prior_audit=dict(cross8=str(CROSS/'completion_audit.json'),cross8_sha256=sha(CROSS/'completion_audit.json'),quality_v3=str(V3/'completion_audit.json'),quality_v3_sha256=sha(V3/'completion_audit.json')),
        code_freeze='IMPLEMENTATION.json before first optimization; evaluation/CPU preparation may run earlier',no_new_frontend_models_or_downloads=True)
    save(NEW/'PLAN.json',plan);(NEW/'plan_hash.txt').write_text(sha(NEW/'PLAN.json')+'\n')
    save(NEW/'ANCHOR_AUDIT.json',dict(state='manifest_and_selected_complete_anchors_passed',old_full_input_audit_reused=str(CROSS/'research/input_audit.json'),selected_snapshot_files_checked=['N04_fused','N06_fused','N11_fused','N04_A','N06_A','N11_A'],manifest_hashes={l:a['input']['sha256'] for l,a in anchors.items()},ROI_generator_sha256=sha(ROOT/'scripts/prepare_diagnostics.py'),LR_source_and_prefix_verified=True))
    save(NEW/'STATUS.json',dict(state='preparing_candidates_and_alpha_evaluation',pid=None,jobs={}))
    print('Frozen plan and anchor audit ready',flush=True)
if __name__=='__main__':main()
