"""Build transparent five-column comparisons and trajectory-level summaries."""
import argparse,asyncio,csv,hashlib,html,json,statistics,os
from pathlib import Path
R=Path(__file__).resolve().parents[1];B=['ftgs_base5k','ftgs_opt15k','4dgs_grid25','4dgs_grid61'];KEYS=['full_psnr','ssim','foreground_psnr','crop_lpips','silhouette_iou','outside_alpha_mean']
def save(p,v):
 t=p.with_suffix(p.suffix+f'.writing.{os.getpid()}');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
from comparison_output import publish, comparison_ready

def summarize():
 selection=json.loads((R/'selection.json').read_text());allrows=[];available={}
 review_path=R/'research/source_action_review.json';review=json.loads(review_path.read_text()) if review_path.exists() else {'cases':{}}
 if review.get('selection_sha256_after'):assert hashlib.sha256((R/'selection.json').read_bytes()).hexdigest()==review['selection_sha256_after']
 for c in selection['main']+selection['reserve']:
  label=c['sample_id'];available[label]={}
  for b in B:
   path=R/'samples'/label/'evaluations'/b/'results.json'
   if path.exists():
    result=json.loads(path.read_text());available[label][b]=result['mean'];action=review['cases'].get(label,{})
    if action:assert hashlib.sha256((R/action['source_contact']).read_bytes()).hexdigest()==action['source_contact_sha256']
    allrows.append({'sample_id':label,'cohort':c['cohort'],'branch':b,'human_action':action.get('category',c['human_action']),'human_action_frozen':c['human_action'],'ego_group':c['ego_group'],'frames':c['output_slots'],**result['mean']})
 stats={}
 for cohort in ['main','reserve']:
  group=[x for x in allrows if x['cohort']==cohort];cohort_stats={}
  for b in B:
   rows=[x for x in group if x['branch']==b];cohort_stats[b]={'available_tracks':len(rows),'mean':{k:statistics.mean(x[k] for x in rows) if rows else None for k in KEYS},'median':{k:statistics.median(x[k] for x in rows) if rows else None for k in KEYS}}
  pairs={}
  for other in B[1:]:
   ids=[c['sample_id'] for c in selection['main']+selection['reserve'] if c['cohort']==cohort and 'ftgs_base5k' in available[c['sample_id']] and other in available[c['sample_id']]];pairs[other]={'paired_tracks':len(ids),'sample_ids':ids,'metrics':{}}
   for k in KEYS:
    delta=[available[i][other][k]-available[i]['ftgs_base5k'][k] for i in ids];sign=-1 if k in ['crop_lpips','outside_alpha_mean'] else 1;pairs[other]['metrics'][k]={'mean_change':statistics.mean(delta) if delta else None,'improved':sum(d*sign>0 for d in delta),'degraded':sum(d*sign<0 for d in delta)}
  grid_ids=[c['sample_id'] for c in selection['main']+selection['reserve'] if c['cohort']==cohort and all(b in available[c['sample_id']] for b in B[2:])]
  grid_metrics={}
  for k in KEYS:
   deltas={i:available[i]['4dgs_grid61'][k]-available[i]['4dgs_grid25'][k] for i in grid_ids};sign=-1 if k in ['crop_lpips','outside_alpha_mean'] else 1
   grid_metrics[k]={'per_track_change':deltas,'mean_change':statistics.mean(deltas.values()) if deltas else None,'median_change':statistics.median(deltas.values()) if deltas else None,'improved':sum(d*sign>0 for d in deltas.values()),'degraded':sum(d*sign<0 for d in deltas.values())}
  stats[cohort]={'branches':cohort_stats,'paired_to_ftgs_base5k':pairs,'grid61_minus_grid25':{'paired_tracks':len(grid_ids),'sample_ids':grid_ids,'metrics':grid_metrics},'groups':{field:{str(v):{'tracks':len(set(x['sample_id'] for x in group if x[field]==v)),'rows':[x for x in group if x[field]==v]} for v in set(x[field] for x in group)} for field in ['human_action','ego_group','frames']}}
 from trajectory_statistics import extend_summary
 denominators=extend_summary(R,selection,allrows,stats)
 summary={'reference':'generated-heldout reconstruction, not realWaymo novel-view accuracy','per_trajectory_first':True,'action_grouping':{'source':str(review_path),'retrospective_visual_description':True,'official_action_annotation':False,'sampling_unchanged':True},'cohort_denominators':denominators,'rows':allrows,'cohorts':stats}
 from budget_statistics import collect as collect_budgets
 summary['budget_comparison']=collect_budgets(R,selection)
 save(R/'metrics_summary.json',summary)
 csv_temp=R/f'metrics.csv.writing.{os.getpid()}'
 with csv_temp.open('w') as f:
  w=csv.DictWriter(f,fieldnames=['sample_id','cohort','branch','human_action','human_action_frozen','ego_group','frames',*KEYS]);w.writeheader();w.writerows(allrows)
 csv_temp.replace(R/'metrics.csv')
 entries=[];unused=[]
 for c in selection['main']+selection['reserve']:
  label=c['sample_id'];base=R/'samples'/label;links=[]
  if not (base/'source/native.json').exists():unused.append(label);continue
  legal=(base/'generation/research/generated_validation.json').exists();gate=base/'source/research/batch_gate_mask.json';failed=gate.exists() and json.loads(gate.read_text()).get('passed') is False and not legal
  for b in B:
   folder=base/'evaluations'/b;p=folder/'results.json'
   status=(f'<a href="../samples/{label}/evaluations/{b}/fixed_action.mp4">完整动作</a> · <a href="../samples/{label}/evaluations/{b}/orbit_fixed_time.mp4">固定时刻环绕</a> · <a href="../samples/{label}/evaluations/{b}/results.json">指标</a> · <a href="../samples/{label}/evaluations/{b}/asset_manifest.json">动态资产清单</a>' if p.exists() else '共同前端失败，依赖阶段未运行' if failed else '等待该分支训练/评测完成')
   links.append(f'<li>{b}: {status}</li>')
  comparisons=''.join(f'<p>生成留出相机 {v}：参考生成视频｜FTGS原5k｜FTGS优化15k｜grid25｜grid61<video controls preload="none" width="100%" src="../samples/{label}/comparison/heldout_view{v:02d}_five.mp4"></video></p>' for v in [0,8,16] if comparison_ready(base))
  sources=[]
  for name,title in [('source/native_contact.jpg','真实Waymo观测'),('source/visualization/observed_crop.mp4','原始观测裁剪视频'),('source/inputs/main.mp4','实际输入视频'),('source/visualization/source_chain.mp4','实拍→输入→骨架→生成'),('source/motion_pair/projection_comparison.mp4','运动估计叠加'),('generation/preflight/conditioning/goliath40/00.mp4','root-local骨架'),('generation/outputs/turbo/videos','24路生成视频')]:
   if (base/name).exists():sources.append(f'<a href="../samples/{label}/{name}">{title}</a>')
  source='<p>'+' · '.join(sources)+'</p>'
  failure='<p class="failure">共同前景/严重遮挡检查未通过，失败记录保留，没有用另一窗口替换该人物。</p>' if failed else ''
  if (base/'comparison/details/manifest.json').exists():source+=f'<p><a href="../samples/{label}/comparison/details/manifest.json">统一手脚代理区域对照清单</a></p>'
  entries.append(f'<section><h2>{label} {c["cohort"]} scene{c["scene"]} cam{c["camera_index"]}</h2><p>{html.escape(c["track_id"])}; {c["native_frame_count"]}张实拍 → {c["output_slots"]}个时间槽；动作描述：{html.escape(review["cases"].get(label,{}).get("category","unknown"))}（回顾性实拍检查，非官方行为标签）</p>{failure}{source}<ul>'+''.join(links)+f'</ul>{comparisons}</section>')
 gallery_temp=R/'gallery'/f'index.html.writing.{os.getpid()}';gallery_temp.write_text('<!doctype html><meta charset="utf-8"><title>Random8 四组动态重建对照</title><style>body{max-width:1500px;margin:auto;font:16px sans-serif;background:#eee}section{background:white;padding:16px;margin:18px 0}video{max-height:660px}.failure{color:#a21}</style><h1>Random8 × 四组动态重建配方</h1><p>Generated-heldout reconstruction。原始实拍与生成相机未配准，不是真实Waymo新视角GT。人工外观批准pending，production_approved=false。只有四组主评测齐全后才发布五栏对照。</p><p>只展示实际尝试的11条轨迹；合法进入后端8人，三条前端失败保留。未启用备用：'+', '.join(unused)+'.</p><p><a href="../REPORT.md">详细报告</a> · <a href="../PROGRESS_LATEST.md">最新进度</a> · <a href="../metrics.csv">终态指标表</a> · <a href="../metrics_summary.json">配对、分组与预算统计</a></p>'+''.join(entries))
 gallery_temp.replace(R/'gallery/index.html')
async def main():
 parser=argparse.ArgumentParser();parser.add_argument('--label');arg=parser.parse_args()
 if arg.label:await publish(R,arg.label)
 summarize()
if __name__=='__main__':asyncio.run(main())
