from pathlib import Path
import json,hashlib,time
R=Path(__file__).resolve().parents[1]
notes=[
'FLUX 条件的 B0/F 改变前格栅和色调；真实条件路线更接近源图，车顶和细节仍存在生成痕迹。',
'六路线保留皮卡类别；B0/F 偏亮，真实条件更接近源色。精修图显示 R 前脸较平滑，F 仍存在车顶及窗框伪影。',
'B0/F 的车身与前脸偏离源车；真实条件路线更接近源车的窗线和色调。',
'B0/F 改变车轮和车身比例；真实条件路线更接近源图外观。',
'B0/F 色调和轮圈有漂移；真实条件路线的车身轮廓更接近源图。',
'B0/F 格栅与车灯更像生成模板；R/H/B0_real4 保留较深车色和源图前脸特征。',
'真实条件更接近深色源图；H 的朝向审核见过尾部前脸重复纹理，不因源视角得分较高而忽略。',
'真实图主要覆盖后侧；R 未见车头生成存在前后相似尾部纹理，heading 置信度 0.5。保留主表并另作九身份敏感性分析。',
'真实图主要覆盖后侧，所有路线未见真实车头均缺乏验证依据。',
'FLUX 将电动 SUV 改为带燃油车格栅的外观，B0/F 继承明显身份漂移；R/H/B0_real4 较接近源车，但存在车顶伪影。'
]
cases=json.loads((R/'inputs/random10_cases.json').read_text());quality=dict(review='Manual image inspection of fixed-source comparison and heading review sheets; no production bank admission',cases=[dict(case_id=c['case_id'],observation=n) for c,n in zip(cases,notes)],limitations=['No ground-truth unseen-side image for every identity','Appearance metrics do not establish geometric accuracy','New batch scene insertion not verified'])
(R/'docs/random10_quality_notes.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2))
index=[];uids=set()
for matrix in ['random10_experiment_matrix.json','random10_additional_matrix.json']:
 tasks=json.loads((R/'inputs'/matrix).read_text())
 for t in tasks:
  run=R/'results'/t['case_id']/f"{t['variant']}_seed{t['seed']}";asset=run/'asset';meta=json.loads((asset/'metadata.json').read_text());uid=meta['uid'];assert uid not in uids;uids.add(uid)
  for file in ['canonical.ply','metadata.json','library.json','turntable.mp4','heading_review.json']:assert (asset/file).is_file()
  for file in ['clip_metrics.json','dino_metrics.json','render_audit.json']:assert (run/'evaluation'/file).is_file()
  t['status']='evaluated_and_deliverable_audited'
  c=json.loads((run/'evaluation/clip_metrics.json').read_text());d=json.loads((run/'evaluation/dino_metrics.json').read_text())
  index.append(dict(uid=uid,case_id=t['case_id'],variant=t['variant'],seed=t['seed'],canonical_ply=str(asset/'canonical.ply'),library=str(asset/'library.json'),metadata=str(asset/'metadata.json'),video=str(asset/'turntable.mp4'),evaluation=str(run/'evaluation'),CLIP_I16=c['I16'],DINO_I16=d['I16'],ED_R=d['ED_R'],production_approved=False,scene_insertion_verified=False))
  # Record review scope explicitly; keep admission status 'review'.
  for name in ['metadata.json','library.json']:
   p=asset/name;data=json.loads(p.read_text());entry=data if name=='metadata.json' else data['assets'][0]
   entry['visual_review_record']=str(R/'docs/random10_quality_notes.json')
   entry['defects']=['Generative appearance or geometry artifacts may remain; see visual review record','New batch scene insertion not verified']
   p.write_text(json.dumps(data,ensure_ascii=False,indent=2))
  if t['variant'] in ['R_refined','F_refined']:
   for name in ['inference.json','status.json']:
    p=run/name;data=json.loads(p.read_text());data['generation_stage_status']=data['status'];data['status']='refinement_export_render_metrics_audit_complete';p.write_text(json.dumps(data,indent=2))
 (R/'inputs'/matrix).write_text(json.dumps(tasks,indent=2))
assert len(index)==70
(R/'docs/random10_asset_index.json').write_text(json.dumps(dict(n_assets=70,n_identities=10,assets=index),indent=2))
files=list((R/'scripts').glob('*.py'))+list((R/'inputs').glob('random10*.json'))+[R/'docs'/p for p in ['random10_evaluation_protocol.json','clip_weights.json','dinov3_weights.json','random10_metrics.json','random10_additional_metrics.json']]
record=dict(created_unix=time.time(),root=str(R),sha256={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},environments=dict(generation=str(R/'env/bin/python'),refinement=str(R/'env_refine/bin/python'),encoders='/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python',renderer='/home/wmj/miniconda3/envs/sim/bin/python'),audit_files=['docs/random10_deliverable_audit.json','docs/random10_additional_audit.json'],note='Input source hashes and model checkpoint hashes are nested in the included query/protocol/model provenance files. No claim of bitwise generation determinism.')
(R/'docs/random10_provenance.json').write_text(json.dumps(record,indent=2))
print('PACKAGED',len(index),'unique assets')
