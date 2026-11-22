"""Local, factual batch status and review gallery; no final-ready auto approval."""
from pathlib import Path
import csv,html,json,datetime
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text()) if p.exists() else None
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
decisions={
 'P13518524':('input_prepared','街道行走；首段 QWH0001 因 alpha 失败拒绝，另选 2.4–7.2s 为 QWH0004'),
 'P5320110':('input_prepared','朝镜头行走；选择 1.8–6.6s，后段截脚不使用；实际下载文件25fps'),
 'P8480544':('rejected','全片只有上半身，抬手出画，缺少腿脚'),
 'P8865815':('rejected_current_static_camera_route','明显跟拍/相机摇动，暂不送入近似静态相机链路'),
 'P8639654':('rejected','人物下肢持续在画外，并有跟拍'),
 'C_ARAKAWA':('rejected','主体较小、横摇、多人物，干净连续目标不足'),
 'UBC01':('research_input_prepared','站立后转身；全身可见；单视频公开再分发许可未明确，保留研究候选'),
 'UBC02':('held','全身转身，但跟踪存在漏帧；当前不插值框、不送生成；单视频许可未明确'),
}
records=[];pages=[]
for p in sorted((ROOT/'sources').glob('*_download.json')):
 r=read(p);sid=r['source_id'];r['source_review_status'],r['review_reason']=decisions[sid]
 if sid=='P13518524':r['author']='utopia 36'
 if sid.startswith('P'):r['license_url']='https://www.pexels.com/license/'
 if sid=='C_ARAKAWA':r['license_url']='https://creativecommons.org/licenses/by/4.0/'
 probe=read(ROOT/'reviews'/sid/'probe.json');r['actual_media']=probe;records.append(r)
 pages.append(f'<article><h2>{html.escape(sid)} · {html.escape(r["source_review_status"])}</h2><p>{html.escape(r["review_reason"])}</p><p><a href="{html.escape(r["source_page"])}">来源页面</a> · {html.escape(str(probe))}</p><video controls preload="metadata" src="raw/{Path(r["path"]).name}"></video><details><summary>全原片抽样图（不等于验收）</summary>')
 for image in sorted((ROOT/'reviews'/sid).glob('contact_*.jpg')):pages.append(f'<img loading="lazy" src="{image.relative_to(ROOT)}">')
 pages.append('</details></article>')
(ROOT/'source_provenance.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
with (ROOT/'source_candidates.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['source_id','source_page','author','license','path','sha256','status','source_review_status','review_reason']);w.writeheader();w.writerows({k:r.get(k) for k in w.fieldnames} for r in records)
assets=[]
for s in read(ROOT/'SELECTED_TRACKS.json')['tracks']:
 aid=s['id'];d=ROOT/'inputs'/aid
 v=dict(asset_id=aid,request_id=None,dataset='web',qualitative_only=True,source_id=s['source_id'],source_files={'original':s['source_path'],'crop':str(d/'crop_unsegmented.mp4'),'frame_mapping':str(d/'inputs/frame_mapping.json')},mask=str(d/'segmentation/target_masks.npz'),
       source_provenance=next(r for r in records if r['source_id']==s['source_id']),payload=None,camera=str(d/'research/camera_interface.json'),timestamps=str(d/'inputs/frame_mapping.json'),
       validation_status=dict(final_figure_ready=False,quality_approved=False,input_mask_gate=read(d/'research/batch_gate_mask.json'),input_visual_review=read(d/'research/input_visual_review.json'),
       temporal=read(ROOT/'reviews'/aid/'pose_temporal_check.json'),conditioning=read(ROOT/'logs'/f'conditioning_{aid}_process.json'),
       generation=read(ROOT/'logs'/f'generate_{aid}_process.json')))
 assets.append(v)
 if (d/'source_crop.mp4').exists():
  pages.append(f'<article><h2>{aid} · {html.escape(s["action"])}</h2><p>原片区间 {s["start_time"]:.3f}–{s["start_time"]+4.8:.3f}s；121帧；当前仅输入审核，尚非最终人体资产。</p>')
  for file,label in [('crop_unsegmented.mp4','真实源裁剪'),('source_crop.mp4','分割输入'),('pose_preview.mp4','原画幅姿态叠加')]:
   if (d/file).exists():pages.append(f'<figure><figcaption>{label}</figcaption><video controls preload="metadata" src="inputs/{aid}/{file}"></video></figure>')
  pages.append(f'<details><summary>全部121帧检查证据</summary>')
  for p in sorted((ROOT/'reviews'/aid).glob('*.jpg')):pages.append(f'<img loading="lazy" src="{p.relative_to(ROOT)}">')
  pages.append('</details></article>')
save(ROOT/'selected_clips.json',dict(clips=read(ROOT/'SELECTED_TRACKS.json')['tracks'],time_protocol='121 unique original-frame samples nearest t0+i/25; no interpolation, loop, duplicate or speed change'))
save(ROOT/'per_asset_validation.json',assets)
summary=dict(schema_version=1,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner='web_human_generation_goal',run_root=str(ROOT),assets=assets,new_final_ready_count=0,existing_camera_owner='appendix_supplementation_conversation',frozen_benchmark_modified=False)
save(ROOT/'web_human_ready_assets.json',summary)
handoff=Path('/mnt/ojc/workplace1/vehicle_model_comparison/handoff/asset_appendix_and_gallery_20261116')
save(handoff/'web_human_generation_progress.json',dict(updated_at=summary['updated_at'],run_root=str(ROOT),source_download_count=len(records),input_review=str(ROOT/'input_review.html'),child_manifest=str(ROOT/'web_human_ready_assets.json'),new_final_ready_count=0,stage='input_review_and_conditioning',existing_camera_task_not_duplicated=True))
with (ROOT/'failures.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['id','stage','reason'])
 for r in records:
  if r['source_review_status'].startswith(('rejected','held')):w.writerow([r['source_id'],'source_screen',r['review_reason']])
 for a in assets:
  gate=a['validation_status']['input_mask_gate']
  if gate and not gate['passed']:w.writerow([a['asset_id'],'SAM2 alpha','Numeric foreground gate failed; retained for diagnosis, not generated'])
(ROOT/'input_review.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>网络人体输入审核</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto;color:#172331;background:#f5f7fa}article{background:white;padding:24px;margin:20px 0;border:1px solid #ddd}video{max-width:340px;max-height:430px}figure{display:inline-block;vertical-align:top;margin:10px}img{max-width:100%}p{line-height:1.65}</style><h1>网络视频人体：真实输入与逐帧审核</h1><p>8个独立下载来源。原片、时间戳、分割和骨架均可追溯。此页只记录候选输入；最终4DGS生成、独立重载和源相机验收分别计数。</p>'+''.join(pages)+'</html>')
print('published',ROOT/'input_review.html')
