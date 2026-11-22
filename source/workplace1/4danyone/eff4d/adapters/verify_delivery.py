"""Artifact-backed final checks; fail rather than silently downgrade a delivery claim."""
import json,datetime,subprocess,sys,os,ast,shutil
from pathlib import Path
from data import ROOT,save_json,sha
checks=[]
def check(name,condition,evidence):
 if not condition:raise AssertionError(name)
 checks.append({'requirement':name,'state':'verified','evidence':evidence})
pre=json.loads((ROOT/'research/preflight/result.json').read_text());check('Native4D CUDA/KNN, camera projection, dynamic gradients and new-process reload',pre['state']=='passed' and pre['projection_max_px']<.005 and pre['replay']['max_abs']==0,'research/preflight/result.json')
for oid in ['OBJ_0018','OBJ_0019']:
 clip=ROOT/'objects'/oid/'clips/C001';m=json.loads((clip/'manifest.json').read_text());check(f'{oid} source24x61, true2.4s and25 originals',len(m['cameras'])==24 and len(m['render_timestamps_sec'])==61 and m['render_timestamps_sec'][-1]==2.4 and m['source_real_frame_count']==25 and m['direct_source_slots']==13 and m['source_interpolated_slots']==48,str(clip/'manifest.json'))
 prot=m['protocols']['S1'];check(f'{oid} split and no init time/view leakage',len(prot['train_camera_ids'])==12 and prot['train_frame_ids']==list(range(0,61,2)) and not(set(prot['train_camera_ids'])&set(m['validation_camera_ids'])) and set(prot['init_frame_ids'])<=set(prot['train_frame_ids']),str(clip/'05_reconstruction/init_S1.json'))
 for path,h in m['source_hashes'].items():checkpath=sha(path);assert checkpath==h,path
 check(f'{oid} source hashes still unchanged',True,f'{len(m["source_hashes"])} files rehashed; manifest.json')
 completed={};config={}
 for v in ['E0','E1']:
  run=clip/'05_reconstruction'/f'{v}_S1_12k_v1';completed[v]=json.loads((run/'complete.json').read_text());config[v]=json.loads((run/'config.json').read_text());c=completed[v];metrics=json.loads((run/'evaluation/metrics.json').read_text());replay=json.loads((run/'replay.json').read_text());asset=clip/'06_assets'/v
  check(f'{oid} {v} actual12k training and48k samples',c['steps']==12000 and c['rendered_samples']==48000 and c['state'].startswith('trained_and'),str(run/'complete.json'))
  check(f'{oid} {v} complete dynamic model asset',all((asset/f).is_file() for f in ['checkpoint.pt','model_spec.json','config.json','replay.json','manifest.json','root_motion.npz']) and sha(asset/'checkpoint.pt')==sha(run/'checkpoint.pt'),str(asset))
  check(f'{oid} {v} fresh-process time replay',replay['new_process'] and len(replay['queries'])==12 and max(q['max_abs'] for q in replay['queries'])==0,str(run/'replay.json'))
  check(f'{oid} {v} all61x3 unweighted evaluation',len(metrics['rows'])==183 and metrics['views']==[0,8,16] and metrics['protocol']['confidence_exclusion'] is False and sum(r['supervised_time'] for r in metrics['rows'])==93,str(run/'evaluation/metrics.json'))
  check(f'{oid} {v} actual authorized GPU',c['device']['physical_gpu'] in [4,5,6,7] and c['device']['cuda_uuid'].startswith('GPU-'),str(run/'complete.json'))
  check(f'{oid} {v} no diffusion and no hidden density change',config[v]['diffusion_weights_loaded'] is False and not config[v]['densification'] and not config[v]['pruning'] and all(config[v][k]==0 for k in ['lambda_sd','lambda_zero123','lambda_svd']),str(run/'config.json'))
 check(f'{oid} paired init and sample sequence/coverage',completed['E0']['initial_parameter_hashes']==completed['E1']['initial_parameter_hashes'] and completed['E0']['sample_sequence_sha256']==completed['E1']['sample_sequence_sha256'] and completed['E0']['observation_counts']==completed['E1']['observation_counts'],str(clip/'08_evaluation/comparison.json'))
 check(f'{oid} actual confidence loss/gradient consumed',completed['E1']['confidence_gradient_test']['feature_gradient_max_change']>0,str(clip/'05_reconstruction/E1_S1_12k_v1/confidence_loss_gradient_test.json'))
 confidence=json.loads((clip/'05_reconstruction/confidence_S1/summary.json').read_text());check(f'{oid} confidence generated from allowed observation neighborhood',confidence['state']=='complete' and len(confidence['views'])==12 and all(v['view'] in prot['train_camera_ids'] and all(all(x['left'] in prot['train_frame_ids'] and x['right'] in prot['train_frame_ids'] for x in f['neighbors']) for f in v['frames']) for v in confidence['views']),str(clip/'05_reconstruction/confidence_S1/summary.json'))
 for view in confidence['views']:assert sha(view['path'])==view['sha256']
 package=json.loads((clip/'09_records/packaging.json').read_text());check(f'{oid} all6 demo categories present and source hashes verified',package['state']=='complete' and package['source_hashes_rechecked'] and all((clip/'07_demo'/name).is_file() for name in ['01_pipeline.mp4','02_backend_comparison.mp4','03_motion_fixed_views.mp4','04_geometry_orbit.mp4','05_details.mp4','06_confidence.mp4']),str(clip/'09_records/packaging.json'))
 for video in package['videos']:assert sha(video['path'])==video['sha256']
 cmp=json.loads((clip/'08_evaluation/comparison.json').read_text());check(f'{oid} baseline comparison boundary explicit','reference' in cmp['B0']['role'] and cmp['production_approved'] is False,str(clip/'08_evaluation/comparison.json'))
 review=json.loads((clip/'08_evaluation/visual_review.json').read_text());check(f'{oid} sampled visual review and unapproved status',review['production_approved'] is False and len(review['sampled_images'])>=3,str(clip/'08_evaluation/visual_review.json'))
 standalone=json.loads((clip/'06_assets/E1/standalone_motion.json').read_text());check(f'{oid} asset-only replay without target images',standalone['state']=='standalone_replay_complete' and standalone['target_images_read'] is False and standalone['video']['frames']==61,str(clip/'06_assets/E1/standalone_motion.json'))
for name in ['README.md','PLAN.json','STATUS.json','REPORT.md','HANDOFF.md','SOURCE_AUDIT.md','publish_manifest.json','gallery/index.html']:
 check(f'Required artifact {name}',(ROOT/name).is_file(),name)
check('Source and environment independence recorded',Path('/mnt/ojc/miniconda3/envs/eff4d/bin/python').is_file() and (ROOT/'scheduler/env.sh').is_file() and (ROOT/'research/runtime_copy.json').is_file(),'scheduler/env.sh; research/runtime_copy.json; research/python_archives.json')
check('Output paths not symlinked into old workspace',not any(p.is_symlink() for p in (ROOT/'objects').rglob('*')),'All object output paths traversed; no symlinks')
check('Disk reserve>=35GiB',shutil.disk_usage(ROOT).free>=35*1024**3,f'FreeGiB={shutil.disk_usage(ROOT).free/2**30:.2f}')
plan=json.loads((ROOT/'PLAN.json').read_text());check('Budget frozen from measured smoke',plan['state']=='S1_frozen_before_formal_training' and plan['formal_steps']==12000,'PLAN.json')
# Formal schedule is the same across objects.
a=json.loads((ROOT/'objects/OBJ_0018/clips/C001/05_reconstruction/E0_S1_12k_v1/config.json').read_text());b=json.loads((ROOT/'objects/OBJ_0019/clips/C001/05_reconstruction/E0_S1_12k_v1/config.json').read_text());keys=['steps','accumulation','microbatch','lr','sampling','initial_spatial_sigma','initial_temporal_sigma_model','densification','background','rgb_loss','alpha_loss'];check('M02 uses frozen R07 recipe',all(a[k]==b[k] for k in keys),keys)
interface=json.loads((ROOT/'research/interface_test/result.json').read_text())
check('Flexible view/time/window/batch interface executed',interface['state']=='passed' and interface['training_views']==7 and interface['render_frames']==31 and interface['rendered_training_samples']==40,'research/interface_test/result.json')
result={'state':'delivery_verified_quality_not_approved','updated_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'checks':checks,'conditional_not_executed':{'E2':'Considered and not selected; current artifacts indicate geometric anisotropy issue, no Zero123/SDS execution claimed','matched_B0_retraining':'Not requested as mandatory; B0 explicitly unequal observation budget reference'},'limitations':['No production visual approval','No real novel-view GT','No independent final test','No strict-resume CLI','No arbitrary-motion retargeting or accurate road/world registration'],'source_hashes_rechecked_at_close':True}
save_json(ROOT/'research/completion_audit.json',result);print(len(checks),'verified checks',flush=True)
