"""Current-source report; separate historical receipts from live work and repaired inputs."""
from pathlib import Path
from datetime import datetime,timezone
import json,asyncio,shutil
from comparison import summarize
from live_inventory import main as observe,alive
R=Path(__file__).resolve().parents[1];BRANCHES=['ftgs_base5k','ftgs_opt15k','4dgs_grid25','4dgs_grid61']
def read(path,default=None):return json.loads(path.read_text()) if path.exists() else default

def archive_actual_receipts():
 directory=R/'logs/actual_receipts';directory.mkdir(exist_ok=True)
 for source in ['STATUS.json','FIRST4_STATUS.json']:
  state=read(R/source,{})
  for key,job in state.get('jobs',{}).items():
   if job.get('pid') and job.get('exit_code') is not None:
    path=directory/(key.replace('/','_')+'_pid'+str(job['pid'])+'_exit'+str(job['exit_code'])+'.json')
    if not path.exists():path.write_text(json.dumps({'source':source,'job':key,'actual_receipt':job},indent=2)+'\n')

def priority_state(state):
 p=read(R/'FIRST4_STATUS.json',{})
 if p:
  state['samples'].update(p.get('samples',{}))
  for key,value in p.get('jobs',{}).items():
   previous=state['jobs'].get(key,{})
   if previous.get('exit_code') is not None and value.get('exit_code') is None and value.get('state','').startswith('reconciled_'):continue
   state['jobs'][key]=value
 return state

def priority_inventory():
 live=observe();p=read(R/'FIRST4_STATUS.json',{});rows={x['job']:x for x in live['jobs']}
 for key,job in p.get('jobs',{}).items():
  rows[key]={'job':key,'receipt_state':job['state'],'state':job['state'],'pid':job.get('pid'),'observed_process_alive':alive(job.get('pid',0)),'log':job.get('log')}
 live['jobs']=list(rows.values());live['priority_controller_pid']=p.get('controller_pid');live['priority_controller_alive']=alive(p.get('controller_pid',0))
 temporary=R/'LIVE.priority.writing';temporary.write_text(json.dumps(live,indent=2)+'\n');temporary.replace(R/'LIVE.json')
 return live

def main():
 archive_actual_receipts();summarize();live=priority_inventory();state=priority_state(read(R/'STATUS.json'));sel=read(R/'selection.json');repair=read(R/'FRONTEND_REPAIR_STATUS.json',{'cases':{}});single=read(R/'SINGLE_GPU_PROFILE_STATUS.json',{});matrix=[];lines=['# Random8：Waymo → 4DAnyone → 四组动态高斯对照','',f'更新时间：{datetime.now(timezone.utc).isoformat()}。目标8条新轨迹、每条4组，完整时32个终态版本。当前仍未全部完成。','', '[详细进度快照](PROGRESS_LATEST.md) · [前四条并发说明](FIRST4.md) · [现场进程](LIVE.json) · [主状态](STATUS.json) · [输入修复](FRONTEND_REPAIR_STATUS.json) · [单卡验证](SINGLE_GPU_PROFILE_STATUS.json) · [Gallery](gallery/index.html) · [指标CSV](metrics.csv) · [配对汇总](metrics_summary.json)','', '本轮只使用Waymo→4DAnyone公开Turbo权重→独立FreeTimeGsVanilla/HUST4DGaussians。没有切换到Efficient4D或IDOL，没有训练4DAnyone前端，没有追加seed、损失或其他研究分支。','', '## 两项审查与工程修复','', '[code_audit.md](code_audit.md) 与 [audit.json](audit.json)：时间f/(F−1)直接传入HexPlane；grid_sample为align_corners=True、padding=border，保留[0,1]约定。grid25实际使用索引12–24，grid61使用30–60。61槽对应2.4秒，121槽4.8秒；有效节点间距分别为0.2/0.08秒与0.4/0.16秒。离散二阶时间平滑权重0.01没有按物理秒数归一化。','', 'coarse关闭形变但实际抽取全时间，zerostamp_init=False、dataloader=False；t0规范初始化不等于coarse只看t0。两阶段H={0,8,16}更新为0。保留这些配方约定，不新增coarse消融。','', '后续核查证明部分旧“前端失败”来自适配器，已用同一人物、同一冻结窗口恢复：','', '- 去畸变直接映射到A@K裁剪，避免固定画布边界切掉仍在原图中的像素。N04全部25帧与原裁剪逐像素相同。','- 640项顶点关键点回归仅将乘加归约改为float64，再转回原float32；权重、SMPL顶点和1e-4刚体门限均未变。三个真实样本最大投影变化0.00235像素，N01/N06恢复通过。','- 曝光末端不足名义时长0.08–0.15毫秒的片段，使用冻结1毫秒容差内的极轻微均匀放慢，保持25fps、61/121槽；不加速、不重复末帧、不补缺失观测。逐槽保留真实源时间。','', '证据：[裁剪探针](research/crop_canvas_probe.json)、[坐标精度探针](research/geometry_precision_probe.json)、[v1.2协议](research/frontend_v12_protocol.json)、[隔离补丁](source/frontend_v12.patch)。N04已生成的v1.1视频保持原样，不重新生成；其他受影响人物使用明确记录的v1.2修复。同一人四个后端始终共享同一份输入。','', '官方固定版本已提供单卡24视角路径，详见[冻结执行配置](research/single_gpu24_profile.json)。先验证N01全部24路视频，再用于后续人物；保持权重、seed42、Turbo四步、sdpa、视角/帧数/分辨率不变。该路径避免四卡模式的跨进程特征临时文件，执行配置变化明确记录，没有自动换模型。','', '## 冻结轨迹与真实阶段','', 'seed=20261015；输入资格池700条、202缓存场景。按官方2D–3D关联去重，无法关联时只声明相机内轨迹。历史跨相机排除核查通过；没有按生成效果选窗口。它不是Waymo全量无筛选随机样本。','', '|ID|组|val scene/cam|实拍→槽|当前阶段|','|---|---|---|---|---|']
 for c in sel['main']+sel['reserve']:
  label=c['sample_id'];sample=state['samples'].get(label,{});ready=repair['cases'].get(label,{});proof=R/'samples'/label/'generation/research/generated_validation.json';stage=sample.get('state','备用未启用')
  if proof.exists():stage='24路生成已验收；'+stage
  elif ready.get('state')=='same_input_repaired_ready_for_first24view_generation':stage='同一冻结输入已修复；等待首次生成'
  elif ready.get('state')=='failed_after_bounded_repair':stage='工程修复后仍未通过原前端检查'
  if label=='N01' and single.get('state','').startswith('running'):stage='GPU6官方单卡24路首次生成验证中'
  genjob=state['jobs'].get(label+'/generate24',{})
  if genjob.get('state')=='running' and alive(genjob.get('pid',0)):stage='GPU'+str(genjob.get('physical_gpus'))+' 正在首次生成24路'
  lines.append(f"|{label}|{c['cohort']}|{c['scene']}/{c['camera_index']}|{c['native_frame_count']}→{c['output_slots']}|{stage}|")
 lines+=['','完整track ID、原生窗口、自车运动、像素高度及选择概率见selection.json。主样本和备用分别计分母，已修复的工程失败不会冒充新人物；只有共同前端硬失败才启用有序备用，最多16条。N09修复时间适配后仍未通过原前景门限，保留失败。动作和附件不强制配额；N04实拍可见行走，源3D动作、附件重建和其他人物动作仍待复核。','', '## 四组配方与已存在的终态指标','', '|分支|冻结配方|','|---|---|','|B0 ftgs_base5k|实际历史原5k配置、零初速度、初始sigma15/(F−1)，独立冷启动5k完整LR计划|','|B1 ftgs_opt15k|实际历史优化配置、零初速度、初始sigma8/(F−1)，独立冷启动15k，保存5k/10k/15k|','|B2 4dgs_grid25|t0训练视角规范轮廓初始化，grid25，3k coarse＋30k fine|','|B3 4dgs_grid61|与B2相同，仅时间网格61；每个人都运行两网格，不按动作挑配方|','', 'HUST原实现最后fine一轮不执行optimizer.step：名义33k轮采样/反传对应32999次实际更新，行为未改。总15k保存点是fine12k；原生pointcloud在当次更新之前保存，因此其参数代表14999次更新，而之后的恢复checkpoint代表15000次。相关预算不能不加说明地称为严格同更新次数。','', '|人物/分支|PSNR↑|SSIM↑|前景PSNR↑|裁剪LPIPS↓|IoU↑|mask外alpha↓|','|---|---:|---:|---:|---:|---:|---:|']
 final_count=0
 for c in sel['main']+sel['reserve']:
  label=c['sample_id']
  for b in BRANCHES:
   path=R/'samples'/label/'evaluations'/b/'results.json'
   if not path.exists():continue
   d=read(path);m=d['mean'];final_count+=1;lines.append(f"|{label}/{b}|{m['full_psnr']:.4f}|{m['ssim']:.6f}|{m['foreground_psnr']:.4f}|{m['crop_lpips']:.6f}|{m['silhouette_iou']:.6f}|{m['outside_alpha_mean']:.6f}|")
 lines+=['',f'当前存在且已验收的终态评测：**{final_count}个版本**。这些是352×640、灰背景0.5、三个H生成留出视角×13时刻的generated-heldout reconstruction指标，不是真实Waymo新视角精度。FTGS5k与15k是完整配方对照，不能把差异单独归因为训练次数或sigma。','', '每个终态评测同时输出H三视角完整61/121时刻的IoU、前景覆盖、外部alpha及下部35%代理区统计；代理区不等于语义脚部指标。原始渲染用于算指标，不从MP4取数，不用目标mask抹掉预测的错误前景。LPIPS使用相同本地AlexNet权重，两环境同图像对数值一致；SSIM沿用相同官方实现。','', '## 后台现场与资源约束','',f"控制器PID={state['controller_pid']}，现场存活={alive(state['controller_pid'])}，状态={state['state']}。tmux socket=`fdwaymo`，session=`random8_r20261015_113332`。切换控制器发生在原训练和评测均退出且没有存活子任务时；没有中断GPU训练。v2曾因缺少监控函数在启动时退出，未启动GPU子任务；已修复并以controller_v3.py运行，通过全局依赖核查。历史失败日志保留。",'']
 p=read(R/'FIRST4_STATUS.json',{})
 if p:lines.append(f"- 前四条优先队列：PID {p.get('controller_pid')}，现场存活={alive(p.get('controller_pid',0))}，状态={p.get('state')}；N01–N03使用新增授权空闲GPU1/6/7，N04由原队列继续。实际子进程退出码备份在logs/first4_receipts。")
 for row in live['jobs']:
  if row['observed_process_alive'] and row['receipt_state']=='running':lines.append(f"- `{row['job']}`：现场PID {row['pid']}；日志 `{row.get('log')}`。")
 if single.get('state','').startswith('running'):lines.append(f"- N01单卡验证：PID {single.get('pid')}，现场存活={alive(single.get('pid',0))}，GPU6；日志 `{single.get('log')}`。")
 for row in live['jobs']:
  if row['state']=='blocked_disk_admission':lines.append(f"- `{row['job']}`：磁盘准入等待，尚未开始训练。")
 for x in live['progress']:
  if x['observed_process_alive']:lines.append(f"- 训练心跳 `{x['file']}`：{x.get('stage','FTGS')} / {x.get('stage_iteration',x.get('step'))}；实际更新{x['actual_optimizer_updates']}。")
 lines+=['',f'磁盘可用 **{shutil.disk_usage(R).free/1024**3:.2f} GiB**，用户要求保留至少35 GiB；调度及保护预留更小的写入缓冲。完整8人32版本按首条估算约需新增30 GiB；以实时余量准入。空间不足停止新大任务准入，不删除旧成果。单卡视频生成可继续推进轻量输出，后端checkpoint的空间约束仍存在。','', 'N01单卡61槽24路已完整通过，实测约699秒；N04实测：24路四卡生成376秒、RGBA98.5秒、FTGS5k111秒、FTGS15k305秒、grid25全训练1813秒。单卡24路吞吐按N01实测更新，不能拿四卡吞吐承诺总时间。当前整批没有可靠ETA，取决于空间和真实输入检查。','', '## 产物入口与剩余工作','', '[Gallery](gallery/index.html) · [逐分支资源/配置审计](research/package_audit_status.json) · [N04相机实际CUDA投影核对](research/N04_camera_projection.json)。HUST/FTGS相机投影最大差0.000111像素，共同规范点对齐误差小于5e-8；这不是Waymo世界尺度验收。','', '五栏视频在samples/<ID>/comparison/：生成参考｜FTGS原5k｜FTGS优化15k｜grid25｜grid61。未完成栏标为PENDING。各evaluations/<branch>/有固定相机动作、固定时刻环绕、黑灰白背景、异常局部图、完整帧统计与动态asset_manifest。FTGS保留.ftgs.ply及配置；HUST保留PLY、deformation/table/accum和runtime。独立进程重放通过才算技术验收。','']
 for c in sel['main']+sel['reserve']:
  label=c['sample_id'];p=R/'samples'/label/'source/visualization/manifest.json'
  if p.exists():lines.append(f"- {label}：[实拍视频](samples/{label}/source/visualization/observed_full_camera.mp4) · [实拍裁剪](samples/{label}/source/visualization/observed_crop.mp4) · [输入/骨架/生成链路](samples/{label}/source/visualization/source_chain.mp4)。实拍为真实25/49观测，链路预览中的实拍列按来源帧保持显示，与生成相机未配准。")
 lines+=['', '剩余必需工作：完成单卡24路验收并接续同一冻结名单 → 存储允许时落盘同一份RGBA/初始化 → 每人完整四组优化 → 动态包独立加载/重放 → 全H时序及六项指标 → 采样序列/配置对照、五栏视频和按轨迹汇总。主8和有序备用单列，不把“准备好”“挂后台”写成已完成资产。','', '生成质量、重建视觉、技术完成和场景就绪分别记录，人工复核pending；production_approved=false。世界尺度、地面接触、完整场景融合和任意SMPL-X换动作均未验证。','', '```bash','tmux -L fdwaymo attach -t random8_r20261015_113332',f'cat {R}/LIVE.json',f'cat {R}/SINGLE_GPU_PROFILE_STATUS.json',f'tail -n 10 {R}/logs/controller_v3.log','```']
 temporary=R/'REPORT.writing';temporary.write_text('\n'.join(lines)+'\n');temporary.replace(R/'REPORT.md')
async def watch():
 while True:
  try:main()
  except Exception as e:print(f'{type(e).__name__}:{e}',flush=True)
  if read(R/'STATUS.json')['state'].startswith('all_registered_tasks_terminal'):return
  await asyncio.sleep(120)
if __name__=='__main__':
 import sys
 asyncio.run(watch()) if '--watch' in sys.argv else main()
