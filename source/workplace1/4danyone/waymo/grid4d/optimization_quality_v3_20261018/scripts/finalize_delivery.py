"""Close the bounded quality round without claiming an unqualified quality success."""
import json,hashlib,re
from pathlib import Path
from datetime import datetime,timezone
NEW=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
    protocol=read(NEW/'research/protocol_audit.json'); recovery=read(NEW/'research/recovery_audit.json')
    assert protocol['state']=='passed' and recovery['state']=='passed_for_all_eleven_complete_runs'
    visual=dict(state='sampled_review_complete_unseen_scope_unknown',method='Direct visual inspection of saved comparison images; full videos decoded separately, not claimed exhaustively viewed in motion.',reviewed=[
        dict(family='N01_A_fusion',files=['parts_rgb.jpg','view00_contact.jpg'],scope='Historical six fixed body crops and view00 frames0/30/60',finding='A and fusion similar at inspected scales; hair/face and shirt lettering remain blurred. HUST hair/leg artifacts reduced in A/fused.',new_missing_limb_in_inspected_frames=False,full_video_flicker='unknown'),
        dict(family='N07_A_fusion',files=['parts_rgb.jpg','view08_contact.jpg'],scope='Historical six fixed body crops and view08 frames0/60/120',finding='Fusion hand/face details worse than A in fixed crops; black holes/streaks visible in hand. Generated reference itself has a head protrusion at inspected mid-time; not all artifacts originate in reconstruction.',full_video_flicker='unknown'),
        dict(family='N04_boundary',files=['parts_rgb.jpg','view16_contact.jpg'],scope='Historical six fixed body crops and view16 frames0/30/60',finding='B-match looks close to A; face/hands remain blurred and walking-foot artifacts persist. No clear large sampled body-loss improvement.',full_video_flicker='unknown'),
        dict(family='N11_color',files=['parts_rgb.jpg','view00_contact.jpg'],scope='Historical six fixed body crops and view00 frames0/60/120',finding='C1 similar to A. C2 adds stronger dark patches/streaks in raised forearm. All retain incorrect magenta skin/arm color and lose clothing pattern; no reliable texture restoration.',new_local_color_damage_C2=True,full_video_flicker='unknown'),
        dict(family='N11_boundary',files=['parts_alpha.jpg'],scope='Historical six fixed alpha body crops',finding='U/B do not eliminate head/arm shape and coverage errors; no new B-match exists.',full_video_flicker='unknown')],global_visual_pass=False,limit='Numerical gates and sampled appearance do not establish complete video quality or real-world identity fidelity.')
    for r in visual['reviewed']:
        r['sha256']={f:sha(NEW/'comparison'/r['family']/f) for f in r['files']}
    save(NEW/'visual_review.json',visual)
    gates=read(NEW/'gate_results.json');gates['state']='fixed_numeric_gates_and_scoped_visual_review_complete';gates['visual_review']='visual_review.json';gates['uniform_new_recipe_accepted']=False
    for r in gates['boundary']+gates['color']+gates['fusion']:
        r['visual']='sampled_only_full_video_unknown'
        if r['candidate']=='C2':r['visual']='sampled_new_color_damage;full_video_unknown'
        r['quality_decision']='rejected_numeric' if not r['numeric_pass'] else 'numeric_pass_only_no_full_visual_acceptance'
    save(NEW/'gate_results.json',gates)
    page=NEW/'comparison/index.html';html=page.read_text().replace('../paired_metrics_partial.csv','../paired_metrics.csv').replace('当前指标 CSV','完整指标 CSV')
    if 'href="quality_time.png"' not in html:html=html.replace('<section><h2>N01_A_fusion','<p><a href="../FINDINGS.md">实验结论</a> · <a href="quality_time.png">固定检查点质量/时间曲线</a></p><section><h2>N01_A_fusion',1)
    page.write_text(html)
    broken=[]
    for ref in re.findall(r'(?:href|src)="([^"]+)"',html):
        if not ref.startswith(('http:','https:','#')) and not (page.parent/ref).exists():broken.append(ref)
    assert not broken,broken
    status=read(NEW/'STATUS.json');assert status['evaluation_counts']=={'complete':41}
    status.update(state='completed_bounded_research_round_no_uniform_new_recipe',at=datetime.now(timezone.utc).isoformat(),quality_accepted=False,quality_decision='A research baseline retained; no uniform replacement from fusion/boundary/color',visual_scope='sampled; unseen full-video behavior unknown',background_jobs_remaining_in_this_directory=False,next_authorized_round='../optimization_cross8_20261018/REPORT.md',completion_audit='completion_audit.json');save(NEW/'STATUS.json',status)
    items=[
      ('Frozen protocol, original inputs and prior reports preserved',['PLAN.json','FUSION_PLAN.json','research/protocol_audit.json']),
      ('N01/N07 fullA, complete recoveries and fixed capacities',['research/recovery_audit.json','ASSETS.json']),
      ('Boundary actual Adam calibration,12 evaluations each, unchanged source states',['update_calibration.json','research/update_calibration/N04_enriched.json','research/update_calibration/N11_enriched.json']),
      ('N04 paired5000 actual samples/LRs/start; N11 not triggered under fixed rule',['research/protocol_audit.json','research/update_calibration/N11.json','source/bmatch.patch']),
      ('Original full boundary protection lines retained, failed outcome reported',['gate_results.json','FINDINGS.md']),
      ('Color native SH/gradient/contribution preflight and official source/license',['color_preflight.json','research/instant_color_source.json','source/instant_color/LICENSE_gaussian_splatting.md']),
      ('C1/C2 fixed geometry/full alpha,20 passes,distinct update budgets',['research/recovery_audit.json','color_comparison.json','ASSETS.json']),
      ('Exact asset/recovery consistency incl fused terminal assets',['research/recovery_audit.json','ASSETS.json']),
      ('All41 independent replay checks incl fractional time and backgrounds',['research/protocol_audit.json']),
      ('Global39 observations, fixed parts/full timeline and denominators',['paired_metrics.csv','comparison/part_metrics_summary.csv','gate_results.json']),
      ('Five complete comparison families,15 fully decoded videos and scoped visual review',['comparison/media_validation.json','visual_review.json','comparison/index.html']),
      ('Fusion fixed recipe, two identities, exact interruption recovery and prefix differences',['FUSION_PLAN.json','research/fusion_migration.json','research/fusion_prefix_comparison.json']),
      ('All28 fixed budget curve points, measured and unmeasured costs distinguished',['efficiency_curve.csv','costs.csv','research/cost_accounting.json','research/step_cost_probe.json']),
      ('Three-person main mean, reserve separate, no false seven-person claim',['comparison/common_cohort.json','PIPELINE_DECISION.md']),
      ('Query design only and independent asset contract',['PIPELINE_DECISION.md']),
      ('No extra original-round seeds/backends/color branches; later user expansion separate',['PLAN.json','FUSION_PLAN.json','NEXT_STEP.md']),
      ('Background queues terminated; preserved failure and repair history',['STATUS.json','EVALUATION_STATUS.json','EVALUATION_REPAIR_STATUS.json','FUSION_STATUS.json','FUSION_RESUME_STATUS.json','README_run.md'])]
    requirements=[]
    for name,paths in items:
        for f in paths:assert (NEW/f).exists(),f
        requirements.append(dict(requirement=name,state='passed_execution_or_documented_conditional_outcome',sources=paths))
    deliver=['PLAN.json','STATUS.json','EVALUATION_STATUS.json','PIPELINE_DECISION.md','FINDINGS.md','paired_metrics.csv','gate_results.json','update_calibration.json','color_preflight.json','color_comparison.json','costs.csv','ASSETS.json','README_run.md','comparison/index.html','NEXT_STEP.md','visual_review.json']
    save(NEW/'completion_audit.json',dict(state='completed_requested_bounded_round',at=datetime.now(timezone.utc).isoformat(),requirements=requirements,counts=dict(requirements=len(requirements),full_replay_snapshots=41,complete_stage_packages=11,complete_videos=15,new_original_branches=5,untriggered_original_branches=1,fusion_addendum_branches=2),candidate_quality='No uniform new candidate accepted. N01 fusion numerical pass is limited single-identity evidence; N07 fails.',quality_gate_success_required_for_experiment_completion=False,limitations=['Unseen full-video visual behavior remains unknown; full decode is not human review.','Shared-load costs do not prove a full-pipeline speedup.','Original first replay-video and CPU part-evaluation wall cost unmeasured; NA.','C1 private sampler and C2 solver matrices reconstituted with explicit provenance; no false runtime or bitwise claim.','Generated H is development data, not real Waymo ground truth or independent final test.'],artifact_hashes={f:sha(NEW/f) for f in deliver},next_user_authorized_scope='../optimization_cross8_20261018/REPORT.md'))
    print('Bounded-round delivery complete; expanded cross8 goal remains active',flush=True)
if __name__=='__main__':main()
