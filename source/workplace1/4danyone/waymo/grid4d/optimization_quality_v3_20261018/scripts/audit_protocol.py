"""Read-only final checks of frozen inputs, actual schedules and replay evidence."""
import json, hashlib, difflib
from pathlib import Path
NEW=Path(__file__).resolve().parents[1]; OLD=NEW.parent/'optimization_boundary_20261018'
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8*1024*1024): h.update(b)
    return h.hexdigest()
def save(p,d): Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def trace(run):
    return [json.loads(l) for p in sorted(run.glob('trace*.jsonl')) for l in p.read_text().splitlines() if l]
def main():
    dependencies=read(NEW/'research/dependencies.json')
    for p,h in dependencies.items(): assert sha(p)==h,p
    plan=read(NEW/'PLAN.json'); fusion=read(NEW/'FUSION_PLAN.json')
    assert sha(NEW/'PLAN.json')==(NEW/'plan_hash.txt').read_text().strip()
    assert sha(NEW/'FUSION_PLAN.json')==(NEW/'fusion_plan_hash.txt').read_text().strip()
    assert sha(NEW/'REQUEST.txt')==plan['request_sha256']
    assert sha(NEW/'USER_STEERING.txt')==plan['steering_sha256']
    for c in plan['cross_person']['configs'].values():
        assert sha(c['path'])==c['sha256']
        s=c['source']
        for k in ['official_recipe','manifest','initialization']: assert sha(s[k])==s[k+'_sha256']
    for c in fusion['configs'].values(): assert sha(c['path'])==c['sha256']
    run=NEW/'runs/N04_Bmatch'; config=read(run/'config.json')
    assert sha(config['source_recovery'])==config['source_recovery_sha256']
    assert sha(NEW/'research/update_calibration/N04.json')==config['calibration_sha256']
    candidate=trace(run); assert len(candidate)==5000
    paired=[]
    for method in ['A','U','B']:
        prior=OLD/'runs/primary'/f'N04_{method}'; oldconfig=read(prior/'config.json'); oldtrace=trace(prior)
        assert len(oldtrace)==len(candidate)
        assert read(prior/'actual_samples.json')==read(run/'actual_samples.json')
        for key in ['source_recovery_sha256','dataset','optimization','pipeline','invariants']:
            assert oldconfig[key]==config[key],(method,key)
        for a,b in zip(candidate,oldtrace):
            for key in ['stage_update','global_actual_updates','view','frame','uid','lr','points']: assert a[key]==b[key],(method,key)
        paired.append(dict(comparator=method,all_5000_samples_LRs_counts_exact=True,common_source=config['source_recovery_sha256']))
    diff=''.join(difflib.unified_diff((OLD/'scripts/train_boundary.py').read_text().splitlines(True),(NEW/'source/train_bmatch.py').read_text().splitlines(True),fromfile='historical train_boundary.py',tofile='quality_v3 train_bmatch.py'))
    (NEW/'source/bmatch.patch').write_text(diff)
    continuations=[]
    for label in ['N01','N07']:
        for stage,budget in [('alpha',15000),('A',5000)]:
            r=NEW/'runs'/f'{label}_{stage}'; t=trace(r); schedule=read(r/'sampling.json'); lrs=read(r/'learning_rates.json')
            assert len(t)==len(schedule)==len(lrs)==budget
            assert [v['uid'] for v in t]==schedule
            assert all(v['learning_rates']==lr for v,lr in zip(t,lrs))
            assert len({v['points'] for v in t})==1
            assert not ({v['view'] for v in t}&{0,8,16})
            continuations.append(dict(run=r.name,steps=budget,sample_LR_trace_exact=True,capacity_fixed=True,H_excluded=True))
    validations=[]
    for p in sorted((NEW/'runs').glob('*/validation/*.json')):
        if p.stem.endswith('_timing'): continue
        d=read(p); assert d['state']=='passed' and d['source_data_reads']==0
        assert d['independent_replay_max_abs_error']<=1e-5
        assert d['background_composition_max_abs_error']<=1e-5
        assert any(v['frame']==12.25 for v in d['time_checks'])
        validations.append(dict(path=str(p.relative_to(NEW)),sha256=sha(p),max_replay_error=d['independent_replay_max_abs_error'],max_composition_error=d['background_composition_max_abs_error']))
    assert len(validations)==41
    metrics=sorted((NEW/'results').glob('*/metrics.json')); assert len(metrics)==41
    media=read(NEW/'comparison/media_validation.json'); assert len(media['families'])==5
    for family in media['families']:
        for video in family['videos']:
            assert sha(NEW/'comparison'/family['name']/video['path'])==video['sha256']
    save(NEW/'research/protocol_audit.json',dict(state='passed',immutable_dependency_files=len(dependencies),plans_unchanged=True,input_manifests_and_initializations_unchanged=True,boundary_comparators=paired,boundary_source_diff='source/bmatch.patch',continuation_checks=continuations,independent_replays=validations,metric_files=len(metrics),complete_media_families=5,decoded_videos=15,interpretation='Protocol and execution checks, not a declaration that every quality gate passed.'))
    print('Frozen protocol, paired schedules,41 independent replays and15 videos passed',flush=True)
if __name__=='__main__': main()
