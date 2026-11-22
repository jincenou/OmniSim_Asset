"""All-input and complete-state audit; no training or rendering updates."""
import csv,hashlib,importlib.util,json
from pathlib import Path
import torch
NEW=Path(__file__).resolve().parents[1];ROOT=NEW.parent;V3=ROOT/'optimization_quality_v3_20261018'
from publish_cross8 import descriptors
spec=importlib.util.spec_from_file_location('recovery_checker',V3/'scripts/audit_assets.py');previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
PLAN=json.loads((NEW/'PLAN.json').read_text())
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def check_sources():
    assert sha(NEW/'PLAN.json')==(NEW/'plan_hash.txt').read_text().strip()
    assert sha(NEW/'FUSION_PLAN.json')==(NEW/'fusion_plan_hash.txt').read_text().strip()
    for name,digest in PLAN['implementation'].items():assert sha(NEW/'scripts'/name)==digest,name
    for config in PLAN['cross_person']['configs'].values():assert sha(config['path'])==config['sha256']
    fusion=read(NEW/'FUSION_PLAN.json')
    for config in fusion['configs'].values():assert sha(config['path'])==config['sha256']
    assert sha(NEW/'source/train_fused.py')==fusion['patched_source_sha256']
    dependencies=read(V3/'research/dependencies.json')
    for p,h in dependencies.items():assert sha(p)==h,p
    inputs=[]
    for label in PLAN['labels']:
        mpath=ROOT.parent/'random8/r20261015_113332/four_dg/data'/label/'fdanyone.json';m=read(mpath);source=Path(m['source_data']);train=held=0
        for r in m['input_hashes']:
            p=source/'images'/f"cam{r['view']:02d}"/f"{r['frame']:06d}.png";assert sha(p)==r['sha256'],p
            if r['view'] in m['train_views']:train+=1
            else:held+=1
        assert train==21*m['frame_count'] and held==3*m['frame_count']
        inputs.append(dict(label=label,manifest_sha256=sha(mpath),training_files=train,heldout_files=held,frame_count=m['frame_count']))
    for key,d in PLAN['reuse'].items():
        r=Path(d['run']);it=53000 if key.endswith('_A') else 33000
        assert sha(r/'complete.json')==d['complete_sha256']
        assert sha(r/'snapshots'/f'{it}.json')==d['snapshot_sha256']
        assert sha(Path(d['results'])/'metrics.json')==d['metrics_sha256']
    save(NEW/'research/input_audit.json',dict(state='passed',inputs=inputs,plans_unchanged=True,prior_dependencies_unchanged=len(dependencies),reused_assets_unchanged=len(PLAN['reuse'])))
def check_outputs():
    checks=[];increments=[];replays=[];schedule=[];prefix=[]
    for label in PLAN['new_A_labels']:
        prior=None
        for stage,it,budget in [('33k',33000,None),('alpha',48000,15000),('A',53000,5000)]:
            run=NEW/'runs'/f'{label}_{stage}';assert (run/'complete.json').exists();s=torch.load(run/'recovery.pt',map_location='cpu',weights_only=False);checks.append(previous.verify_recovery(run,it,s))
            if prior:
                assert prior['gaussians']['_xyz'].shape==s['gaussians']['_xyz'].shape
                for key in ['gaussian_optimizer','deform_optimizer']:
                    before,after=prior[key],s[key]
                    for g,h in zip(before['param_groups'],after['param_groups']):
                        assert g['name']==h['name']
                        for a,b in zip(g['params'],h['params']):assert float(after['state'][b]['step'])-float(before['state'][a]['step'])==budget
                increments.append(dict(run=run.name,all_group_step_increments=budget,capacity_fixed=True))
                seq=read(run/'sampling.json');lr=read(run/'learning_rates.json');trace=[json.loads(v) for v in (run/'trace.jsonl').read_text().splitlines()];assert len(trace)==len(seq)==len(lr)==budget
                assert [r['uid'] for r in trace]==seq and [r['learning_rates'] for r in trace]==lr
                assert not ({r['view'] for r in trace}&{0,8,16})
                if stage=='A':
                    historical=read(ROOT/'optimization_boundary_20261018/data/N04/learning_rates.json')
                    assert all(all(row[k]==historical[j][k] for k in row) for j,row in enumerate(lr))
                schedule.append(dict(run=run.name,samples_and_LRs_exact=True,H_excluded=True))
            prior=s
        del prior,s
        native=[json.loads(v) for v in (NEW/'runs'/f'{label}_33k/trace.jsonl').read_text().splitlines()]
        fused=[json.loads(v) for v in (NEW/'runs'/f'{label}_fused/trace.jsonl').read_text().splitlines()]
        assert len(native)==len(fused)==33000
        for a,b in zip(native[:15000],fused[:15000]):
            for k in ['iteration','view','frame','current_lr']:assert a[k]==b[k],(label,k)
        prefix.append(dict(label=label,first15000_sampling_and_LR_exact=True,native_points_at15000=native[14999]['points'],fused_points_at15000=fused[14999]['points'],limit='Independent CUDA trajectories, not a bitwise single-factor paired run.'))
    for label in PLAN['new_fused_labels']:
        run=NEW/'runs'/f'{label}_fused';s=torch.load(run/'recovery.pt',map_location='cpu',weights_only=False);checks.append(previous.verify_recovery(run,33000,s));del s
    for run in sorted((NEW/'runs').iterdir()):
        if not (run/'complete.json').exists():continue
        for p in sorted((run/'snapshots').glob('*.json')):
            m=read(p)
            for f,h in m['files'].items():assert sha(run/f)==h
            validation=read(run/'validation'/p.name);assert validation['state']=='passed' and validation['source_data_reads']==0
            assert validation['independent_replay_max_abs_error']<1e-5 and validation['background_composition_max_abs_error']<1e-5
            assert any(c['frame']==12.25 for c in validation['time_checks'])
            replays.append(dict(run=run.name,snapshot=p.stem,max_error=validation['independent_replay_max_abs_error'],composition_error=validation['background_composition_max_abs_error']))
    assert len(checks)==len(replays)==18
    save(NEW/'research/recovery_audit.json',dict(state='passed',exports=checks,optimizer_increments=increments,schedule_checks=schedule,pre_alpha_prefix_checks=prefix,independent_replays=replays,reused_evidence=str(V3/'completion_audit.json'),dynamic_contract='Full asset and matched runtime, not arbitrary action driving'))
    print('All18 new stage/reconstruction exports and recoveries passed',flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--inputs-only',action='store_true');a=p.parse_args();check_sources()
    if not a.inputs_only:check_outputs()
