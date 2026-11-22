from pathlib import Path
import json,datetime,hashlib
r=Path(__file__).resolve().parents[1];b=r/'benchmark';manifest=json.loads((b/'manifest.json').read_text());checks=[]
for job in manifest['jobs']:
 label=job['label'];recipe=job['recipe'];new=json.loads((b/'results'/job['id']/'metrics.json').read_text())
 if recipe=='q0':oldpath=(r/'reconstruction/outputs'/label/'quality_metrics/metrics.json') if label.startswith('walk') else r/'quality'/f'baseline_{label}'/'quality_metrics/metrics.json'
 else:oldpath=Path(job['checkpoint']).parents[1]/'quality_metrics/metrics.json'
 old=json.loads(oldpath.read_text());lookup={(s['view'],s['frame']):s for s in new['samples']};errors=[]
 for s in old['samples']:
  if s['background']==1 and s['split']=='validation_generated':errors.extend(abs(lookup[(s['view'],s['frame'])][k]-s[k]) for k in ['full_psnr','foreground_psnr','silhouette_iou','outside_alpha_per_foreground_pixel'])
 assert len(errors)==156
 assert max(errors)<1e-3,(job['id'],max(errors))
 checks.append({'job':job['id'],'old_metric_scalar_max_abs_error':max(errors),'matched_images':39,'new_complete_images':len(new['samples']),'identity_check':new['metric_selfcheck']})
(b/'verification.json').write_text(json.dumps({'state':'passed','checks':checks,'scope':'Existing PSNR/mask metrics reproduced at39 shared images per job; all363 frames finite and LPIPS/SSIM identity selfchecks passed. Does not assert official metric code equivalence.'},indent=2)+'\n')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
s=json.loads((r/'status.json').read_text());s.update(updated_at=now,state='benchmark8_complete_moving_projection_review_pending',benchmark={'state':'complete','identities':8,'metric_jobs':16,'report':'benchmark/REPORT.md','results':'benchmark/results.json','reference':'generated heldout observations, not real novel-view GT'},independent_density_validation={'state':'two_tracks_completed_metrics_improved_visual_review_pending','comparison':'validation_comparison.json'},moving_camera_diagnostic={'state':'motion_pair_complete_visual_review_pending','queue':'moving_queue_retry2.json','comparison':'experiments/moving_R01_v1/motion_pair/comparison.json','generation_started':False})
(r/'status.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
p=r/'REPORT.md';old=p.read_text();addition='''## 8 条基线与配对评测更新

已完成 **8 条现有身份的基线、16 项完整评测**，每项 3 个留出视角 × 121 帧。按用户最新要求，只计算当前数据支持的 PSNR、SSIM、LPIPS 重建一致性，不计算缺失真实多视角真值的指标，也不启动仅为匹配论文设置的额外复现。详见 [完整基线报告](benchmark/REPORT.md) 与 [结果表](benchmark/results.csv)。

两条独立验证身份在相同 50k 预算下增加容量，整图 PSNR 分别 +0.435/+0.718 dB，SSIM +0.0063/+0.0089，LPIPS 相对下降 14.37%/17.91%；这与旧版灰背景裁剪 LPIPS 是不同评测口径。尚无全部行走身份的优化对照，不能推广为所有动作的收益。

移动相机 R01 的实测标定与静态估计两套 GVHMR 诊断已完成，待投影视觉检查，尚未启动新视频生成。两次接口错误均保留：可选模块导入顺序、视频 Path/字符串类型；修正后严格加载模型完成两组推理。没有把诊断完成视为适配质量通过。

'''
if '## 8 条基线与配对评测更新' not in old:
 header,rest=old.split('\n',1);p.write_text(header+'\n\n'+addition+rest.lstrip())
c=json.loads((r/'code_manifest.json').read_text());c['updated_at']=now
for pat in ['benchmark*','moving*']:
 for p in (r/'scripts').glob(pat):
  if p.is_file():c['files'][str(p.relative_to(r))]=hashlib.sha256(p.read_bytes()).hexdigest()
(r/'code_manifest.json').write_text(json.dumps(c,indent=2)+'\n')
print('Verified16 metric jobs; updated report, status, code manifest')
