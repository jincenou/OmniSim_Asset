"""Evidence-backed delivery manifest for baseline and completed continuation assets."""
import ast
import json
from pathlib import Path
from html.parser import HTMLParser
from baseline_queue import ROOT,save,now
from render_baseline import sha


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        self.links.extend(v for k,v in attrs if k in ['href','src'])


def main():
    evaluation=json.loads((ROOT/'optimization/EVALUATION_STATUS.json').read_text())
    assert evaluation['state']=='complete_quality_requires_visual_review'
    paired=json.loads((ROOT/'optimization/comparison/validation.json').read_text());assert paired['state']=='passed'
    training=json.loads((ROOT/'optimization/STATUS.json').read_text());assert training['state']=='training_replay_complete_quality_pending'
    for path in ['audit/historical_exposure_queue.json','audit/gallery_queue.json','audit/training_queue.json','audit/support_queue.json']:
        assert json.loads((ROOT/path).read_text())['state']=='complete',path
    assets=[]
    for label in ['N04','N11']:
        for method in ['baseline','continue','parts','alpha','combined']:
            run=ROOT/'runs'/f'{label}_33k_v1' if method=='baseline' else ROOT/'optimization/runs'/f'{label}_{method}'
            iteration=33000 if method=='baseline' else 48000
            snapshot=run/'snapshots'/f'{iteration}.json';snap=json.loads(snapshot.read_text())
            validation=run/'validation'/f'{iteration}.json';val=json.loads(validation.read_text());assert val['state']=='passed'
            files=[run/name for name in snap['files']]+[run/'config.json',run/'asset_camera_data.json',snapshot,validation,run/'snapshots'/f'{iteration}_reference.npz']
            for name,digest in snap['files'].items():assert sha(run/name)==digest
            assets.append(dict(label=label,cohort='reserve' if label=='N11' else 'main',method=method,iteration=iteration,actual_updates=snap['actual_optimizer_updates'],points=snap['points'],
                files=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in files],
                recovery=str((run/'recovery.pt').relative_to(ROOT)),standalone_command=f'python scripts/replay_asset.py --run {run} --iteration {iteration} --view 0 --output {ROOT}/tmp/{label}_{method}_replay.mp4',
                independent_replay_max_error=val['independent_replay_max_abs_error'],coordinate_scope='Original normalized generated canonical space; metric world scale and ground contact not established',time_scope='Retained25fps generated timeline; interpolation within clip, not new-action driving'))
    code=list((ROOT/'scripts').glob('*.py'))+list((ROOT/'scripts').glob('*.sh'))
    for p in code:
        if p.suffix=='.py':ast.parse(p.read_text())
    runtime=[ROOT/'research/source.patch',ROOT/'research/source_verification.json',ROOT/'research/environment.lock.txt',ROOT/'research/preflight.json',ROOT/'optimization/PLAN.json']+code
    save(ROOT/'ASSETS.json',dict(state='complete_model_packages_technical_replay_passed',at=now(),assets=assets,
        runtime_files=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in runtime],source_commit='3ab7b74482910d22fba7d51ac60c8cae1fe24bf1',environment='/mnt/ojc/miniconda3/envs/grid4d',
        source_directory='source',source_provenance='All pinned upstream blobs verified; local adaptations in source.patch; this manifest hashes runtime wrappers',
        portability='Keep model package, matching source/extensions/environment and wrapper files. Environment inherits base packages read-only; not a self-contained Docker image.',
        standalone_validation='research/standalone/N04_view01.json',all_other_final_assets_replayed_in_new_process=True))
    linkaudit=[]
    for file in [ROOT/'gallery/index.html',ROOT/'optimization/comparison/index.html']:
        parser=Links();parser.feed(file.read_text());missing=[]
        for link in parser.links:
            if link.startswith(('#','http:','https:','data:')):continue
            if not (file.parent/link.split('#')[0]).exists():missing.append(link)
        assert not missing,(str(file),missing)
        linkaudit.append(dict(path=str(file.relative_to(ROOT)),local_links_checked=len(parser.links),missing=missing))
    evidence=['audit/inventory.json','audit/stage_part_audit.jsonl','audit/part_metrics.csv','audit/paired_summary.csv','audit/part_metric_audit.json','audit/budget_quality.csv','audit/costs.csv','audit/visual_review.json','mechanism/static/results/summary.json','optimization/comparison/validation.json','optimization/alpha_gradient_acceptance.json','ASSETS.json']
    assert all((ROOT/p).is_file() for p in evidence)
    save(ROOT/'completion_audit.json',dict(at=now(),state='technical_delivery_passed_final_quality_interpretation_required',assets=len(assets),original_identities=2,new_identity_count=0,optimized_variants=8,html=linkaudit,
        required_evidence={p:sha(ROOT/p) for p in evidence},training_and_eval_statuses=dict(training='optimization/STATUS.json',evaluation='optimization/EVALUATION_STATUS.json'),
        requirements=dict(A='8x4 diagnostics completed; actual historical sampling audited; unrecorded old gradients/events disclosed',B='Two official pinned Grid4D33k assets with fixed snapshots,replay,quality and cost comparison',C='Four static diagnostics plus user-authorized eight fixed-capacity continuation variants; no claimed C2 error-sampling or C3 structure allocation experiment',D='Conditional trigger not met: donor correspondence/transfer unverified',optional_MEt3R='not_run'),
        limitations=['Single seed; numerical atomic nondeterminism, not bitwise training reproducibility','Generated development reference, not real novel-view ground truth','Conservative core/estimated optical flow; unknown preserved','No matched GPU-time comparison','No claimed source-motion/world registration/new-action driving','Alpha global finite-difference failed result preserved; piecewise acceptance narrowly scoped']))
    print(json.dumps(dict(state='passed',assets=len(assets),links=linkaudit)),flush=True)


if __name__=='__main__':main()
