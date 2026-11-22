from pathlib import Path
import json,csv,sys,numpy as np
R=Path(__file__).resolve().parents[1];O=R/'scene_validation';sys.path.insert(0,str(R/'tools/report_deps'));import markdown
from docx import Document
m=json.loads((O/'docs/summary.json').read_text());audit=json.loads((O/'docs/insertion_audit.json').read_text());plans=json.loads((O/'inputs/scenes.json').read_text());ground=json.loads((O/'docs/grounding_comparison.json').read_text())
def table(h,rows):return '\n'.join(['| '+' | '.join(h)+' |','|'+'|'.join(['---']*len(h))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
tab=table(['路线','完成插入','可见帧数/360','插入前景CLIP ↑','插入前景DINO ↑'],[[r['variant'],'6/6',r['visible_frames'],f"{r['clip']:.4f}",f"{r['dino']:.4f}"] for r in m]);plantab=table(['场景','原模板','最小ego间距m','最小已有车辆间距m'],[[s['scene'],s['maneuver'],f"{s['audited_capsule_gap_ego_m']:.3f}",f"{s['audited_capsule_gap_agents_m']:.3f}"] for s in plans]);gtab=table(['场景','平均接地高度变化m','原平均倾角°','低位路面平均倾角°'],[[r['scene'],f"{r['mean_contact_height_change_m']:.3f}",f"{r['original_mean_tilt_deg']:.3f}",f"{r['lower_road_mean_tilt_deg']:.3f}"] for r in ground])
report=f'''# 车辆资产插入重建场景：三场景同条件验证

已完成3个原生nuScenes重建场景 × 2个对象 × 4条路线，共24组插入视频。每组60帧、12fps、6相机，共8640张相机渲染；另有3个无插入背景参考视频。6个四路线对比视频可直接汇报。

本轮证明了八个现有车辆高斯能通过原SplatAD渲染入口随轨迹插入，并在共用条件下比较。**没有证明所有结果已自然融合或可以直接上线。**三个场景仍能看到悬浮感、缺少接触阴影、颜色与背景光照不匹配；这些共用适配问题不能仅靠换生成模型解决。

## 一、验证对象与实际pipeline

两个对象：`random10_04_white_suv`、`random10_02_white_pickup`；路线B0、R、H、B0_real4，均seed42。这些是之前完成的车辆实验，不是本次新增的10个非车辆。SUV参考长度4.362m、宽2.093m；皮卡参考长度5.738m、宽2.401m，来自之前元数据辅助的同一参考。各路线只按同一长度进行等比缩放，保留各自宽高差异；没有为了好看分别调宽高。

已训练SplatAD checkpoint(step30000) → 原nuScenes地图和场景提取 → 原轨迹模板规划 → 共用包围范围检查 → 资产统一尺度与坐标 → LiDAR/场景Gaussian路面拟合 → 按时间插值的统一接地姿态 → 将资产Gaussians追加到原rasterization输入 → 原深度排序与RGB decoder → 六相机同步视频。

不是把物体图片贴到背景上。所有路线上同一场景使用同一checkpoint、轨迹、相机序列、采样时刻、颜色校准探针和参考接地范围。已验证contact数组和颜色校准逐值相同；不含资产的后向相机块与无插入背景逐像素完全一致，最大差0，排除了缓存复用导致背景条件变化。

{plantab}

轨迹规划前，以8个待插入资产最大水平范围加余量，使用约5.838×2.889m的统一碰撞检查范围。表中为原规划器的胶囊近似间距复核，不是精确三角网格接触真值。每条规划取80个10Hz状态；渲染原相机索引30–89，共约5秒。这里没有使用未来轨迹优化或闭环ego反应。

## 二、接地排查与共用修正

最初三个H短片在原高度网格上出现明显倾斜。邻近路缘、车体等高点会污染局部拟合。采用原工程已有的低位路面处理：邻域网格高度25%分位+0.12m筛选后拟合平面，斜率异常则报错；对所有路线使用相同规则。先在10Hz轨迹上平滑高度和法向，再插值到相机时刻，消除了原来按相机遍历顺序平滑的依赖。增加平移及yaw角速度，使插入资产参与原滚动快门渲染。

{gtab}

上表比较同一SUV完整轨迹的共用接地序列，说明拟合发生了什么，**不是毫米级接地准确率**。没有独立路面真值；画面仍有悬浮感。RGB到场景16维特征使用原有带符号仿射decoder校准，不做每对象颜色拟合；原插入器不生成新接触阴影，也没有真实环境重光照。pitch/roll角速度尚未传入滚动快门模型。

## 三、插入诊断与前景外观

{tab}

“完成插入”只计视频与相机渲染完成。可见帧数表示六相机中至少一个相机的资产可见像素（alpha>0.1）≥64；出画、遮挡和投影面积不同都会影响它，不能直接当质量排名。

插入前景CLIP/DINO是新增的**源外观诊断**：固定第0、30、59帧；只取四路线共同可见且alpha>0.5像素≥64的相机；从实际场景RGB中以资产可见alpha提取前景，独立方形归一化、白底512，再与原固定源前景计算CLS余弦相似度。先在每个“对象×场景”内平均相机/时刻，再平均3场景，最后平均2个对象。编码器与车辆/非车辆I16相同，但视角集合不同，**不是CLIP-I16、DINO-I16或ED-R，不能把这些数与旧表直接比较**。只有2个独立对象，不将6个场景组合当成6个独立身份。

原六相机渲染还记录了逐帧 isolated/visible alpha质量、遮挡比例、边界裁切和前景面积。遮挡比例较低可能表示正确被卡车挡住，并非失败；CSV保留原值而不把它合成为“总质量分”。没有加入物体的真实GT视频，因此不计算相对于空背景的PSNR/SSIM，这种分数会错误地奖励少插入或不插入。

## 四、视觉结论与选择

- B0车辆更宽高、后部细节更清楚，但此两例在等长条件下显得偏大；插入CLIP最高并不代表比例最佳。
- R较保留真实证据，但源视角不足时的背面结构与较粗糙外观仍影响环绕和插入；皮卡在场景中明显偏冷色。
- H在这两个对象的插入前景DINO均值最高，仍有偏色、无接触阴影等问题；没有证据证明它在所有插入质量维度都最好。
- B0+4 real保留原TRELLIS后端，整体表现接近H，是原工程继续接入的合适备选。仍需检查宽高比例、支撑位置和光照。

**研究主线继续保留H，工程低改动候选保留B0+4 real；本轮不作最终生产入库裁决。**如果目标是轨迹渲染自然，下一优先级应是独立校验路面与相机投影、轮胎支撑/尺度、接触阴影与场景颜色适配，再扩展身份和场景。仅提高CLIP/DINO或源图PSNR不足以解决现在最明显的融合问题。

## 五、交付文件

- `results/<场景>/<对象>/comparison_scene.mp4`：4路线同一FRONT相机的5秒对比，采用所有路线/所有帧共同确定的固定裁剪，未逐路线平移缩放。
- `comparison_frame_000/030/059.jpg`：起始、中间、末尾截图。
- `<路线>/insert_splatad_6cam.mp4`：原六相机完整输出，便于核验出画和遮挡。
- `random10_04_white_suv/background/insert_splatad_6cam.mp4`：无插入的背景参考。
- `docs/insertion_diagnostics.csv`、`clip_scene_metrics.json`、`dino_scene_metrics.json`、`insertion_audit.json`：诊断、逐时刻特征分数、共用条件检查。
- `inputs/*_traj.npz`、`*_plan.json`、`scenes.json`：原规划结果和配置；各输出目录保留接地序列、checkpoint兼容审计与sha256。
'''
(O/'REPORT.md').write_text(report);(O/'REPORT.txt').write_text(report);css='body{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1200px;margin:auto;padding:28px;line-height:1.65}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px}img,video{max-width:100%}section{margin:30px 0}'
(O/'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><style>'+css+'</style>'+markdown.markdown(report,extensions=['tables']))
doc=Document();lines=report.splitlines();i=0
while i<len(lines):
 line=lines[i]
 if line.startswith('|'):
  block=[]
  while i<len(lines) and lines[i].startswith('|'):block.append([x.strip() for x in lines[i].strip('|').split('|')]);i+=1
  t=doc.add_table(rows=1,cols=len(block[0]));t.style='Table Grid'
  for j,x in enumerate(block[0]):t.rows[0].cells[j].text=x
  for row in block[2:]:
   cells=t.add_row().cells
   for j,x in enumerate(row):cells[j].text=x
  continue
 if line.startswith('#'):doc.add_heading(line.lstrip('# ').strip(),min(len(line)-len(line.lstrip('#')),3))
 elif line:doc.add_paragraph(line.replace('**','').replace('`',''))
 i+=1
doc.save(O/'REPORT.docx')
html='<!doctype html><meta charset="utf-8"><style>'+css+'</style><h1>三场景车辆插入对比</h1><p><a href="REPORT.html">完整报告</a> · <a href="REPORT.docx">Word</a> · <a href="docs/insertion_diagnostics.csv">诊断CSV</a></p>'+markdown.markdown(tab,extensions=['tables'])+'<p>全部完成不等于自然融合。仍有悬浮感、接触阴影缺失和颜色不匹配。指标口径与I16不同。</p>'
for scene in ['scene-0103','scene-0104','scene-0277']:
 for cid in ['random10_04_white_suv','random10_02_white_pickup']:
  path=f'results/{scene}/{cid}';html+=f'<section><h2>{scene} / {cid}</h2><video controls preload="none" src="{path}/comparison_scene.mp4"></video><img loading="lazy" src="{path}/comparison_frame_030.jpg"></section>'
(O/'START_HERE.html').write_text(html)
with (O/'docs/summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(m[0]));w.writeheader();w.writerows(m)
print('SCENE REPORT COMPLETE',flush=True)
