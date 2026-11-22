"""Requirement-scoped audit of delivered baseline files and prior decoded proofs."""
from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 rows=[];tracks=set()
 for item in read(R/'delivery_audit.json')['records']:
  demo=Path(item['demo']);v=read(demo/'validation.json');job=v['job'];metrics=read(Path(item['metrics']))
  checkpoint=Path(job['checkpoint']);assert sha(checkpoint)==v['checkpoint_sha256']
  asset=read(checkpoint.parents[1]/'asset.json');identity=(str(asset['source_scene']),asset['source_track']);assert identity not in tracks;tracks.add(identity)
  for key,field in [('input','input_sha256'),('mapping','mapping_sha256'),('native_mapping','native_mapping_sha256')]:assert sha(Path(job[key]))==v[field]
  for name,video in v['videos'].items():
   assert sha(demo/name)==video['sha256'];assert video['frames']==(72 if name=='orbit_fixed_time.mp4' else 121)
  assert len(list(demo.glob('details_view*.jpg')))==9
  assert metrics['ply_roundtrip']['render_max_abs']<=1e-5
  assert max(metrics['ply_roundtrip']['all_parameter_max_abs'].values())<=1e-6
  assert len([x for x in metrics['samples'] if x['split']=='validation_generated'])==117
  rows.append({'label':item['label'],'checkpoint_and_input_hashes':'matched','demo_video_hashes':'match_full_decode_records','detail_previews':9,'metric_views':[0,8,16],'times':13,'backgrounds':3,'ply_max_render_error':metrics['ply_roundtrip']['render_max_abs'],'technical':True,'final_visual_approved':False})
 controls=[]
 for row in read(R/'quality_results.json')['rows']:
  root=Path(row['metrics']).parents[1];s=read(root/'sampling_step4999.json');counts=s['counts_by_camera_and_frame']
  assert s['optimized_steps']==5000 and sum(sum(x) for x in counts)==5000
  assert all(sum(counts[v])==0 for v in [0,8,16]);assert len(s['train_camera_ids'])==row['training_views']
  controls.append({'label':row['label'],'views':row['training_views'],'optimizer_updates':5000,'holdout_updates':0})
 walk=R/'reconstruction/outputs/walk063_rootlocal24';proof=read(walk/'demo/replay_validation.json');geometry=read(R/'experiments/walk063_rootlocal24_v1/research/root_local.json')
 assert all(v['all_joints_all_frames_inside'] for v in geometry['coverage'])
 assert max(v['joint_translation_render_max_abs'] for v in proof['translation_checks'])<1e-5
 assert geometry['root_checks']['root_relative_joint_vectors_max_abs']<1e-6
 assert geometry['root_checks']['vertices_world_vertical_max_abs']==0
 assert len(read(walk/'demo/details/validation.json')['records'])==15
 result={'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'baseline_delivery_verified_current_experiments_incomplete','previous_goal_turn_classification':'progress','baseline12':rows,'strict_view_controls':controls,'walking':{'root_local_geometry':'checked','views_all_time_inside_margin':24,'root_replay_equivalence':'passed','details':15,'full_final_visual_approval':False},'completed_requirements':['12distinct baseline trajectories and retained failures','same-camera generated-vs-GS demos','common-six initialization view-count experiments','two-track5k/15k/native50k/capacity controls','two additional standing validation tracks','root-local walking generation, GS, original/new trajectory replay','expanded generated validation metrics and8-track whole-frame baseline','local gallery, status, reports, patches, commands and device evidence'],'remaining':['Four mobility controls still need terminal training/evaluation/video evidence and visual comparison','Short runtime21-frame real GPU generation and visual review pending','Moving-camera branch remains diagnostic; not accepted for generation','More walking/action diversity remains limited; do not label standing aswalking'],'not_required_or_not_claimed':['100assets this round','all13assets passing final appearance quality','real novel-view PSNR withoutGT','arbitrarySMPL-Xretargeting','Waymo metric scene insertion'],'no_goal_completion_claim':True}
 (R/'delivery_requirement_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'baseline12':len(rows),'controls':len(controls),'state':result['state']}))
if __name__=='__main__':main()
