"""Final read-only artifact verification; no optimization or model selection.

Existing tensor/runtime probes are accepted only when their source artifacts match
current hashes. Visual unknowns are preserved; execution success is not quality.
"""
from pathlib import Path
import json,csv,hashlib,math,datetime,itertools,zipfile
import numpy as np
B=Path(__file__).resolve().parents[1];ROOT=B.parent
BASE=ROOT.parent/'random8/r20261015_113332'
def read(p):return json.loads(Path(p).read_text())
def rows(p):
 with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def verify(p,h):assert sha(p)==h,str(p)
def finite(x):
 if isinstance(x,float):assert math.isfinite(x)
 elif isinstance(x,dict):
  for v in x.values():finite(v)
 elif isinstance(x,list):
  for v in x:finite(v)
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-8,abs_tol=1e-10),(a,b)
def record(name,proof,detail):checks.append(dict(requirement=name,status='verified',evidence=proof,scope=detail))
plan=read(B/'PLAN.json');checks=[];files={}
assert plan['labels']==['N04','N06','N11'] and plan['arms']==['R0','R1','R2','R3']
verify(B/'research/task.txt',plan['task_sha256'])
assert read(B/'research/freeze.json')['formal_runs_at_freeze']==0
regs=plan['training']['new_regularizers']
assert all(regs[k]==1e-4 for k in ['position_normalized_square','color_normalized_square','offset_normalized_square'])
assert regs['position_color_before_stage_gate'] and regs['once_per_dual_view_update'] and regs['offset_increment_only']
assert plan['gates']['effectiveness']['appearance']['expanded_RGB_MSE_ratio_max']==.95
assert plan['gates']['effectiveness']['completeness']['expanded_soft_alpha_undercoverage_ratio_max']==.9
for rel,h in plan['implementation_sha256'].items():verify(B/rel,h)
for rel,h in plan['evaluation']['expanded_ROI_sha256'].items():verify(B/rel,h)
record('Scope, user numeric corrections and frozen implementation/ROI','PLAN.json; research/freeze.json; research/user_clarification.json','Three people, four branches; preregistered coefficients and thresholds; original plan plus logged resource/border amendments, no quality scan')
manifest=read(B/'research/runtime_manifest.json')
for rel,h in manifest['native_source_files'].items():verify(ROOT/'source'/rel,h)
for rel,h in manifest['shared_helpers_sha256'].items():verify(rel,h)
for rel,h in manifest['adapter_runtime_sha256'].items():verify(B/rel,h)
record('Native implementation and matching runtime identities','research/runtime_manifest.json; research/renderer_audit.json','Actual source hashes checked; private final-world/SH override only; source commit is not a claim of pristine upstream checkout')
pre=read(B/'research/preflight.json');assert pre['passed'] and pre['technical_updates']==24<=200
for f,h in pre['proofs'].items():verify(B/f,h)
probe=read(B/'research/probe/report.json')
assert probe['native_RGB_max_abs']==probe['native_world_max_abs']==0
assert probe['full_state_restore_exact'] and probe['continuous_bake_exact'] and probe['pre_gate_regularizer_verified']
assert probe['null_all_invalid_weight']==1
assert all(v['gradient']>0 and v['change']>0 for v in probe['gradient_and_parameter_updates'].values())
assert read(B/'research/projection_border_test.json')['passed'] and read(B/'research/gate_test.json')['passed']
record('Real differentiable branch, zero-head identity, topology and restore probes','research/preflight.json; research/probe/report.json; research/control_probe.json; research/projection_border_test.json','24 technical updates excluded from quality; Q/K/V/offset/head active after head update; local native raster gradients, not global smoothness')
audit=read(B/'research/completed_training_audit.json');assert audit['training_audit_complete'] and not audit['pending'] and len(audit['runs'])==15
recoveries=[];metrics_count=0;terminal_count=0
for label in plan['labels']:
 inp=plan['inputs'][label];T=inp['frame_count']
 for path,hashkey in [('manifest','manifest_sha256'),('initialization','initialization_sha256'),('anchor_file','anchor_sha256')]:verify(inp[path],inp[hashkey])
 verify(B/'data'/label/'schedule.json',inp['schedule_sha256'])
 schedule=read(B/'data'/label/'schedule.json');assert len(schedule)==32999
 cams=read(B/'data'/label/'cameras.json');assert len(cams['train_views'])==21 and not set(cams['train_views'])&{0,8,16}
 assert len(inp['support_views'])==4 and set(inp['support_views'])<=set(cams['train_views'])
 for t,v1,v2 in schedule:assert 0<=t<T and v1!=v2 and {v1,v2}<=set(cams['train_views'])
 with np.load(inp['initialization']) as init,np.load(inp['anchor_file']) as a:
  xyz=init['points'];close(np.linalg.norm(xyz.max(0)-xyz.min(0)),inp['L'])
  assert a['anchors'].shape==(128,3)
  assert all(np.any(np.all(xyz==x,axis=1)) for x in a['anchors'])
 baseline=read(BASE/'samples'/label/'evaluations/4dgs_grid61/results.json')
 signature=sorted((r['view'],r['frame'],r['crop_xyxy'],r['input_sha256']) for r in baseline['samples'])
 for r in baseline['samples']:verify(BASE/'samples'/label/'data/images'/f"cam{r['view']:02d}"/f"{r['frame']:06d}.png",r['input_sha256'])
 for arm in ['warmup']+plan['arms']:
  key=f'{label}_{arm}';run=B/'runs'/key;proof=audit['runs'][key]
  expected=[2999] if arm=='warmup' else [5000,15000,18000,32999]
  assert [r['step'] for r in proof['recoveries']]==expected and proof['passed']
  for rec in proof['recoveries']:
   p=Path(rec['path']);verify(p,rec['sha256']);assert p.stat().st_size==rec['logical_bytes']
   recoveries.append({'file':str(p.relative_to(B)),'sha256':rec['sha256'],'bytes':p.stat().st_size})
  trace=[json.loads(s) for s in (run/'trace.jsonl').read_text().splitlines()]
  assert trace[-1]['step']==expected[-1] and all([r['frame'],*r['views']]==schedule[r['step']-1] for r in trace)
  done=read(run/'training.complete.json');assert done['step']==expected[-1] and done['counts']['updates']==expected[-1]
  if arm!='warmup':
   terminal_count+=1
   resume=read(run/'resume.json');assert Path(resume['source'])==B/'runs'/f'{label}_warmup/recovery_2999.pt'
   assert resume['sha256']==audit['runs'][f'{label}_warmup']['recoveries'][0]['sha256']
   assert done['counts']['target_renders']==65998 and done['counts']['alpha_renders']==35998
  for step in expected:
   m=read(B/'results'/f'{key}_{step}/metrics.json');finite(m);assert m['actual_updates']==step and len(m['samples'])==39
   assert sorted((r['view'],r['frame'],r['crop_xyxy'],r['input_sha256']) for r in m['samples'])==signature
   for field,val in m['mean'].items():close(val,sum(r[field] for r in m['samples'])/39)
   metrics_count+=1
  print('verified recoveries and metrics',key,flush=True)
assert len(recoveries)==51 and metrics_count==51 and terminal_count==12
record('Cold starts, paired observations, no H training inputs, all full curves','runs/*/resume.json; data/*/schedule.json; scripts/runtime.py; scripts/common.py','All 32999 scheduled exposures checked; actual logged samples match; cold provenance and init anchors checked; training input loader plus audit guard excludes H')
record('51 immutable full recovery points and 51 stage evaluations','research/completed_training_audit.json; runs/*/recovery_*.pt; results/*/metrics.json','Every recovery freshly rehashed; tensor/optimizer checks linked to exact same bytes; all means and 39 H observation/crop/input hashes independently recomputed')
asset_rows=[]
for label,arm in itertools.product(plan['labels'],plan['arms']):
 key=f'{label}_{arm}';asset=B/'assets'/key;meta=read(asset/'asset.json');ex=read(asset/'export_validation.json');rep=read(asset/'replay_validation.json')
 for f,h in meta['files'].items():verify(asset/f,h)
 assert meta['actual_step']==32999 and ex['passed'] and ex['raw_PLY_parameters_exact'] and ex['base_hash_deform_exact']
 assert ex['recovery_sha256']==audit['runs'][key]['recoveries'][-1]['sha256']
 if arm!='R0':assert ex['query_exact']
 assert rep['passed'] and rep['original_Random8_read_guard_active'] and rep['same_3d_state_reused_for_all_cameras']
 assert rep['fractional_definition_fixed'] and rep['tolerance']==1e-5 and len(rep['rows'])==36
 T=plan['inputs'][label]['frame_count'];coords={(r['frame'],r['view'],r['background']) for r in rep['rows']}
 assert coords==set(itertools.product([0.,12.25,30.5,float(T-1)],[0,8,16],[0.,.5,1.]))
 assert all(max(r['max_abs'],r['composition_max_abs'],r['alpha_max_abs'])<=1e-5 for r in rep['rows'])
 partrows=rows(B/'results'/f'{key}_32999/parts.csv');assert len(partrows)==T*3*6
 rawproof=read(B/'cache/raw'/f'{key}_32999/complete.json');assert rawproof['frames']==T and rawproof['views']==[0,8,16]
 for f,h in rawproof['files'].items():verify(B/'cache/raw'/f'{key}_32999'/f,h)
 asset_rows.append({'key':key,'points':meta['points'],'replay_checks':36,'export_exact':True,'max_replay_error':max(r['max_abs'] for r in rep['rows'])})
record('12 standalone assets and dense full-time six-part results','ASSETS.json; assets/*/{asset,export_validation,replay_validation}.json; results/*/parts.csv','All asset/raw-render bytes match recorded hashes; 432 new-process replay cases at required views/times/backgrounds; no original-image replay and identical continuous-time definition checked in runtime source')
for label in plan['labels']:
 m=read(B/'research'/f'{label}_terminal_mechanism.json');finite(m)
 assert m['parameter_updates']==0 and m['gradient_observed_for_all_required_parameters'] and m['perturbation_reaches_3d_residual_all_sampled_times']
 assert m['baked_Z_all_times']['shape']==[plan['inputs'][label]['frame_count'],128,64] and m['baked_Z_all_times']['dtype']=='torch.float32'
 assert m['baked_Z_all_times']['finite'] and m['baked_Z_all_times']['temporal_std_mean']>0
 assert all(r['baked_integer_max_abs']==0 and r['all_candidates_invalid_null_weight']==1 for r in m['rows'])
 assert all(v>0 for v in audit['runs'][label+'_R3']['query_parameter_max_change_5000_to_32999'].values())
record('Formal Query/offset updates, support perturbation and baked-Z diagnostics','research/*_terminal_mechanism.json; research/completed_training_audit.json','Three actual terminal R3s; input perturbation changes 3D residuals; sample-frame and recorded late-training diagnostics only, no material correspondence or useful-alignment claim')
# Expanded ROI aggregation and reference denominator checks across all seven methods.
for label in ['N06','N11']:
 signature=None
 for method in ['R0','R1','R2','R3','old_A','old_fused','old_HUST']:
  key=label+'_'+method+('_32999' if method.startswith('R') else '')
  rs=rows(B/'results'/key/'expanded.csv');assert len(rs)==6*plan['inputs'][label]['frame_count']
  sig=[tuple(r[k] for k in ['view','frame','side','known','roi_pixels','reference_alpha_mass','temporal_valid_pixels']) for r in rs]
  if signature is None:signature=sig
  else:assert sig==signature,(label,method)
  summary=read(B/'results'/key/'expanded.json')['summary']
  for side in [0,1]:
   selected=[r for r in rs if int(r['side'])==side];valid=[r for r in selected if r['known']=='True'];s=summary[f'side{side}']
   assert len(valid)==s['known_rows'] and len(selected)-len(valid)==s['unknown_rows']
   area=sum(int(r['roi_pixels']) for r in valid);mass=sum(float(r['reference_alpha_mass']) for r in valid)
   close(s['RGB_MSE'],sum(float(r['RGB_error_sum']) for r in valid)/area)
   close(s['soft_undercoverage'],sum(float(r['deficit_sum']) for r in valid)/mass)
   close(s['positive_excess'],sum(float(r['positive_excess_sum']) for r in valid)/mass)
g=read(B/'gate_results.json');assert g['extra_R1_percentage_threshold']==False and g['unknown_not_pass']
assert all(r['identical'] for r in g['denominator_checks'])
for r in g['effectiveness']:
 expected=all(c['state']=='pass' for c in r['checks']);assert expected==r['measured_scope_numeric_pass']
 assert r['numeric_candidate']==(expected and r['unknown_rows']==0)
 for c in r['checks']:
  if c['value'] is not None:close(c['value'],c['candidate']/c['comparator']);assert (c['value']<=c['limit'])==(c['state']=='pass')
record('Original/expanded ROI, primary efficacy, protection, R1 and history comparisons','gate_results.json; expanded_summary.csv; part_summary.csv; scripts/analyze.py; scripts/evaluate_expanded.py','14 full-time expanded evaluations; same reference/temporal denominators across seven methods verified; E/D/excess reaggregated; unknown never passes; no invented R1 percentage gate')
visual=read(B/'visual_review.json');assert visual['all_cases_review_recorded'] and not visual['pending_cases']
for label,entry in visual['cases'].items():
 for f in entry['files']:verify(B/f['path'],f['sha256'])
 assert entry['quality_pass'] is not True
media=[]
for label in plan['labels']:
 d=read(B/'comparison'/label/'complete.json');assert d['all_four_arms_complete'] and len(d['files'])==6
 for r in d['files']:
  verify(B/r['path'],r['sha256']);assert r['fully_decoded'] and r['frames']==plan['inputs'][label]['frame_count'];media.append(r)
record('Complete visual comparisons and honest visual review','comparison/*/complete.json; visual_review.json','18 full-time videos freshly hashed, fully decoded on production and again in downloadable package; 3-view endpoint review all people, all25 frames in two fixed bad windows; other frames remain unknown')
cost=read(B/'cost_summary.json');assert cost['formal_unique_optimizer_calls']==368997 and cost['technical_optimizer_calls']==24
assert len(rows(B/'costs.csv'))==12 and len(rows(B/'stage_metrics.csv'))==48
assert len(read(B/'post_queue.complete.json')['jobs'])==101 and not read(B/'training_queue.complete.json')['failed']
record('Actual exposure, topology, parameter/storage/memory and timing costs','costs.csv; cost_summary.json; COSTS.md; runs/*/snapshot_32999.json','Shared prefix counted once; same 32999 is not equal compute to old fused; loading/save included in entry wall time, shared-GPU and unmeasured components disclosed')
pack=ROOT/'deliverables/4DAnyone_Meeting_20261019.zip';pv=read(pack.with_suffix('.validation.json'));verify(pack,pv['sha256'])
with zipfile.ZipFile(pack) as z:assert z.testzip() is None
assert pv['videos']==66 and pv['images']==144 and pv['local_links']==393
record('User-requested offline meeting package and teacher brief','../deliverables/4DAnyone_Meeting_20261019.zip; ../deliverables/4DAnyone_Meeting_20261019.validation.json; BRIEF.txt','Eight-person pipeline selection plus three-person Query; relative links verified; 66 videos decoded,144 images checked, archive CRC and SHA checked')
required=['PLAN.json','research/mechanisms.md','research/implementation.patch','README_run.md','REPORT.md','FINDINGS.md','NEXT_STEP.md','METRICS.md','ASSETS.json','COSTS.md','visual_review.json']
for rel in required:
 p=B/rel;assert p.is_file() and p.stat().st_size>0;files[rel]=sha(p)
record('Final required narrative and reproducibility deliverables',required,'Findings distinguish implementation success from failed/unknown quality, old A and new protocol, per-person limitations; one bounded evidence-backed next step proposed, not silently launched')
record('Architecture/source review against task module specification','scripts/{evq,runtime,train,render_evq,assets,common}.py; research/renderer_audit.json','Reviewed actual source: 128x64 anchors, two 6-channel scales,4 supportsx4 samplesx2scales,2headQKV,null token,4px offsets,top4 association,2x128MLP,pregate bounded world/DC residuals,zero head,native7D transform,no time-opacity or image teacher; frozen files hashes verified above')
record('Resource authorization, bounded scope and preserved exceptions','research/capacity_probe.json; scripts/{scheduler,post_scheduler,publish_scheduler,storage}.py; queue_history.json; research/storage_amendment.json; research/projection_boundary_amendment.json','GPU UUID binding, memory measurement before queue, asynchronous tmux subprocess admission; no other-task termination; original import failure retained; user relaxed35GiB reserve; no additional people/seeds/front-end training')
result=dict(schema='evq_completion_audit_v1',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution_and_required_deliverables_complete=True,quality_default_approved=False,quality_limitations=['N06 vs R0 primary efficacy fails','N11 vs R2 measured primary efficacy fails','N11 20 unknown left-ROI rows','Full-sequence manual visual review outside specified inspected scope remains unknown','R1/R2 explain most global gain; no statistically validated independent Query advantage'],checks=checks,counts=dict(terminals=12,recovery_points=51,stage_global_evaluations=51,full_time_part_evaluations=12,expanded_evaluations=14,videos=18,new_asset_replay_cases=432),recoveries=recoveries,assets=asset_rows,required_artifact_sha256=files,goal_state_update_requires_review=True)
(B/'completion_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print('COMPLETION AUDIT: verified',len(checks),'requirement groups; quality not approved',flush=True)
