"""Refresh the run report without touching any legacy report or GPU workload."""
from pathlib import Path
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import json,shutil,os,asyncio
from comparison import summarize
R=Path(__file__).resolve().parents[1]
def main():
 from live_inventory import main as observe
 live=observe()
 summarize();selection=json.loads((R/'selection.json').read_text());status=json.loads((R/'STATUS.json').read_text());lines=['# Random8：Waymo → 4DAnyone → 四组动态重建对照','',f"更新时间：{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}。本轮隔离目录 `{R}`。",'','当前状态以 [STATUS.json](STATUS.json) 和现场进程为准。目标8条新轨迹、32个后端终态版本；不代表已完成32个版本。Efficient4D、IDOL和旧实验没有改动。','',f"本轮控制器状态：`{status['state']}`；任务状态计数：`{dict(Counter(v['state'] for v in status['jobs'].values()))}`。",'',f"磁盘可用 {shutil.disk_usage(R).free/1024**3:.2f} GiB，准入保留35 GiB。空间不足的任务会等待，不删除旧成果；完整8人预计还需额外约30–40 GiB，首条实际体积仍在测量。",'', '## 两项审查与隔离修改','', '详情见 [code_audit.md](code_audit.md)、[audit.json](audit.json)。61/121槽的实际时间链均为 `f/(F-1)` → camera.time → deformation → HexPlane，没有改为[-1,1]。grid_sample使用align_corners=True、padding=border；grid25有效整数节点12–24（13个），grid61为30–60（31个）。时间轴为时间平面张量axis2，空间多分辨率不放大时间长度。当前二阶离散平滑不按物理秒数重标定，因此网格对照同时改变容量和离散正则。','', 'coarse关闭形变，但实际抽取全时间，不是只监督首帧；zerostamp_init=False、dataloader=False。规范初始化取t=0训练视角，和coarse抽样是两回事。两阶段H={0,8,16}均排除，已做真实采样、前向和反向探针。没有据此新增coarse消融。','', '隔离改动：训练日志/采样账本、必要保存节点、无损gzip恢复checkpoint、数值异常退出守卫；所有核心损失、学习率、增密、时间约定保持历史配方。前端将名义10Hz帧索引插值改为真实曝光时间括号和比例，统一用于本轮；不补末帧、不改变25fps。见 [曝光时间修正](research/exposure_time_correction.json)。完整来源和patch见source/。','', 'HUST训练器名义3k+30k=33k次采样/反向，但原实现跳过最后fine optimizer.step，实际32999次更新。保持原行为。其point_cloud保存发生在当次optimizer更新之前，恢复checkpoint在之后；总15k保存点对应fine12k。','', '## 冻结抽样','', '筛选池700条、202个缓存场景；抽样seed=20261015，生成/训练seed=42。先按官方2D–3D关联去重，否则仅声明相机内轨迹身份。主8条选中概率8/700；不是Waymo全量无筛选随机样本。按原生像素、遮挡/截断、连续性选定一次窗口，未看输出后换窗口。动作及携带物未人工复核，仍为unknown；移动自车不等于人体行走。','', '|样本|组|scene/cam|track ID|实拍→槽|原生高度中位数|自车平移/旋转|现场状态|','|---|---|---|---|---|---|---|---|']
 for c in selection['main']+selection['reserve']:
  sample=status['samples'].get(c['sample_id'],{});lines.append(f"|{c['sample_id']}|{c['cohort']}|val/{c['scene']}/{c['camera_index']}|{c['track_id']}|{c['native_frame_count']}→{c['output_slots']}|{c['height_median_px']:.1f}px|{c['ego_path_m']:.2f}m / {c['ego_rotation_deg']:.2f}°|{sample.get('state','inactive reserve')}|")
 lines+=['','主样本硬失败仍计入主8分母；仅共同前端硬失败可按备用顺序补充，最多16次尝试。后端失败和外观一般不换人。初始失败详情及日志保留在STATUS.json中，不把不同来源的失败合并成“模型效果差”。','', '## 四组独立配方','', '|分支|初始化与预算|保存节点|','|---|---|---|','|B0 ftgs_base5k|原历史5k配置、零初速度、sigma15/(F−1)，独立冷启动5k完整LR计划|最终5k|','|B1 ftgs_opt15k|历史优化配置、零初速度、sigma8/(F−1)，独立冷启动15k完整LR计划|5k/10k/15k|','|B2 4dgs_grid25|t0训练视角规范轮廓、grid25、coarse3k+fine30k|coarse3k/总15k/最终33k|','|B3 4dgs_grid61|与B2相同，时间网格61；独立模型，独立但同seed采样RNG|同B2|','', '每人只生成一次24路704×1280视频；四组共用352×640 RGBA、相机、时间。H={0,8,16}不参与初始化或训练。B0不能用B1的5k权重替代，B2/B3不互相热启动。训练实际配置、输入SHA、设备UUID、初始化参数摘要及采样轨迹分别保存。','', '## 当前现场与后台队列','', '实际命令、PID、日志和退出码见STATUS.json及logs/*.json。后台使用 `tmux -L fdwaymo`，会话 `random8_r20261015_113332`。最多四个后端进程，四卡生成整体准入；共享其他项目但不终止它们。磁盘不足会显示blocked_disk_admission。','']
 for key,job in status['jobs'].items():
  if job['state'] in ['running','waiting_gpu_memory','blocked_disk_admission']:
   lines.append(f"- `{key}`：{job['state']}；PID={job.get('pid','—')}；GPU={job.get('physical_gpus','—')}；日志 `{job.get('log','—')}`。")
 for directory in list((R/'samples').glob('*/ftgs_*'))+list((R/'four_dg/runs').glob('*')) if (R/'four_dg/runs').exists() else list((R/'samples').glob('*/ftgs_*')):
  path=directory/'progress.json'
  if path.exists():
   data=json.loads(path.read_text());lines.append(f"- `{directory.relative_to(R)}` 最新步数：`{json.dumps(data,ensure_ascii=False)}`。")
 lines+=['', '现场进程以 [LIVE.json](LIVE.json) 为准；历史running但PID不存在会标记stale，不能当作GPU仍在运行。评测修复的成功记录与首次失败分别保留。', '', '当前已成功的终态评测：'+str([(v['label'],v['branch'],v['mean']) for v in live['successful_evaluations'] if v['snapshot'] is None]), '', '## 指标、动态资产与可视化','', '[本轮gallery](gallery/index.html) · [指标CSV](metrics.csv) · [按轨迹汇总](metrics_summary.json)。暂无结果的入口明确标为pending；不把路径计划当作已生成成果。','', '指标从原始渲染计算：352×640、灰背景0.5、三个H视角×13个固定时刻。整图PSNR、SSIM、目标前景PSNR、相同目标mask联合裁剪LPIPS、IoU、mask外平均alpha。完整H视角61/121时刻同时输出IoU、覆盖率、外部alpha和下部35%代理区统计。下部代理不是语义脚部精度。','', '每分支独立进程重新加载动态包，FTGS逐参数PLY读回并重渲染首/中/末；HUST重新加载PLY+deformation/table/accum，与训练结束的浮点参考对比。最终目录 samples/<Nxx>/evaluations/<branch>/ 保存结果、固定相机动作、固定时刻环绕、黑灰白背景及异常局部图；samples/<Nxx>/comparison/ 保存三个H的五联屏。','', '所有数值只代表 generated-heldout reconstruction，不是实拍Waymo新视角精度，也不能证明源动作、世界尺度、地面接触或任意SMPL-X换动作正确。技术、生成质量、重建视觉和场景就绪分别记录；production_approved始终为false，人工复核pending。','', '## 查看与接续','', '```bash', 'tmux -L fdwaymo attach -t random8_r20261015_113332',f'cat {R}/STATUS.json',f'tail -n 12 {R}/logs/controller_attempt4.log','```','', '下一步自动执行：共同24视角生成 → RGBA与训练视角初始化 → 四分支独立优化 → 独立动态重放及评测 → 五联屏与汇总。若资源阻塞，保持后台等待并记录原因，不静默改配方或新建其他实验分支。']
 (R/'REPORT.md').write_text('\n'.join(lines)+'\n')
async def watch():
 while True:
  try:main()
  except Exception as e:print(type(e).__name__+':'+str(e),flush=True)
  s=json.loads((R/'STATUS.json').read_text())
  if s['state'].startswith('all_registered_tasks_terminal'):return
  await asyncio.sleep(120)
if __name__=='__main__':
 import sys
 asyncio.run(watch()) if '--watch' in sys.argv else main()
