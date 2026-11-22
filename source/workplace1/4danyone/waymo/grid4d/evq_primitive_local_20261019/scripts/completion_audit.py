"""Verify all required terminal evidence; quality failures are preserved, not completion failures."""
from pathlib import Path
import json,hashlib,csv,datetime
N=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 plan=read(N/'PLAN.json');checks=[]
 for rel,h in plan['implementation_sha256'].items():assert sha(N/rel)==h
 for path,h in plan['source_integrity'].items():assert sha(Path(path))==h
 runtime=read(N/'research/runtime_manifest.json')
 for rel,h in runtime['native_source_files'].items():assert sha(N.parent/'source'/rel)==h
 for path,h in runtime['shared_helpers_sha256'].items():assert sha(Path(path))==h
 audit=read(N/'research/completed_training_audit.json');assert audit['complete'] and len(audit['runs'])==6
 all_updates=0;files={}
 for label in plan['labels']:
  for arm in plan['arms']:
   key=f'{label}_{arm}';run=N/'runs'/key;asset=N/'assets'/key;result=N/'results'/f'{key}_32999'
   done=read(run/'training.complete.json');assert done['step']==32999 and done['counts']['additional_updates']==30000;all_updates+=done['counts']['additional_updates']
   a=next(x for x in audit['runs'] if x['key']==key);assert a['passed'] and [x['step'] for x in a['snapshots']]==[5000,15000,18000,32999]
   for rec in a['snapshots']:
    p=run/f'recovery_{rec["step"]}.pt';assert p.stat().st_size==rec['bytes'] and read(run/f'snapshot_{rec["step"]}.json')['recovery_sha256']==rec['sha256']
   meta=read(asset/'asset.json')
   for name,h in meta['files'].items():assert sha(asset/name)==h
   replay=read(asset/'replay_validation.json');assert replay['passed'] and replay['original_Random8_read_guard_active'] and replay['same_3d_state_reused_for_all_cameras'] and replay['fractional_definition_fixed'];assert len(replay['rows'])==36
   metric=read(result/'metrics.json');assert len(metric['samples'])==39 and len(metric['dense_samples'])==3*plan['inputs'][label]['frame_count']
   parts=list(csv.DictReader((result/'parts.csv').open()));assert len(parts)==6*3*plan['inputs'][label]['frame_count']
   expanded=read(result/'expanded.json');assert expanded['all_methods_same_masks_denominators'];checks.append(dict(key=key,training=True,snapshots=4,independent_replay=True,global_observations=39,part_rows=len(parts),expanded_unknown_preserved=True))
  media=read(N/'comparison'/label/'complete.json');assert len(media['files'])==6
  for row in media['files']:assert row['fully_decoded'] and sha(N/row['path'])==row['sha256']
 assert all_updates==180000
 visual=read(N/'visual_review.json');assert set(visual['people'])=={'N06','N11'} and visual['scope_explicit'] and visual['unseen_not_pass']
 for key in ['gate_results.json','paired_metrics.csv','part_summary.csv','expanded_summary.csv','last_frame_leg_core.csv','costs.csv','ASSETS.json','METRICS.md','FINDINGS.md','NEXT_STEP.md','research/source.patch','research/runtime_manifest.json','comparison/index.html','README_run.md']:
  p=N/key;assert p.is_file() and p.stat().st_size>0;files[key]=sha(p)
 out=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='six primitive-local reconstruction experiments only; scene20 frontier batch has separate status',execution_complete=True,all_quality_gates_passed=all(g['whole_scope_numeric_candidate'] for g in read(N/'gate_results.json')['personwise']) and bool(visual.get('all_quality_passed',False)),quality_outcomes_source='gate_results.json',total_formal_new_updates=all_updates,terminal_assets=6,full_recovery_snapshots=24,complete_comparison_videos=12,checks=checks,files=files,old_sources_unchanged=True,no_automatic_additional_training=True,visual_scope=visual)
 (N/'completion_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('six experiment execution delivery audited')
if __name__=='__main__':main()
