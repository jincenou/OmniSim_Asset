"""Current project report from saved, verified process and metric evidence."""
from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parents[1]
def read(name):return json.loads((R/name).read_text())
def main():
 snapshot=read('progress_snapshot.json');bench=read('benchmark/results.json');stamp=snapshot['sampled_at_beijing']
 lines=['# 4DAnyone × Waymo 动态资产：当前进度与后续计划','',f'现场核验时间：**{stamp}（北京时间）**。训练步数和预计时间是该时刻快照；后台继续运行。',
 '', '**当前结论：24 视角动态资产链路已跑通，站立样本的重建优化已有量化收益，正在验证这些收益能否迁移到行走和迈步/转身。仍有外观错误、细节模糊和漂浮点，尚未达到稳定批量高质量生产。**',
 '', '[全部效果页](gallery/index.html) · [8 条基线完整指标](benchmark/REPORT.md) · [当前训练状态](mobility_queue.json) · [本次现场核验](progress_snapshot.json)',
 '', '## 一、已经完成到什么程度','',
 '|项目|已完成|应如何理解|','|---|---|---|',
 '|原批次 24 视角动态资产|12 条不同人体轨迹|已有视频、checkpoint、动态 PLY、渲染和数值验收；旧批次已收口|',
 '|新增行走分支|1 条，walk063_rootlocal24|完成 24 视角生成、5k 动态高斯、原根轨迹和新合成轨迹回放|',
 '|本轮 24 视角技术完成总量|13 条轨迹|12 条旧批次＋1 条新行走；不把六视角试验、训练版本或多视角数算成新人物|',
 '|最终综合视觉质量验收|0 条完成|不是“没有结果”；意味着还未批准为高质量生产资产。抽样复核保留了模糊和失败|',
 '|冻结基线评测|8 条轨迹、16 项评测|8 份 5k 基线＋4 条轨迹各自的原容量 50k、增加容量 50k；不是 16 个人|',
 '|视角数量控制|2 条 × 6/12/21 训练视角，共 6 组|共用由六个训练视角得到的初始化，防止多视角初始化泄漏|',
 '|训练预算/分辨率/容量|已完成两条调参轨迹及两条独立轨迹验证|详细结论见下文；独立轨迹未用于选密度配方|',
 '|移动相机 R01|原始标定审计、49→121 插值、两套 GVHMR 推理及投影视频|仅诊断完成；尚无该分支新的生成视频或高斯资产|',
 '|100 条资产|未完成，也未恢复批量提交|仍按用户后续指示，优先少量优化与对照|',
 '', '计数按不同轨迹，不声称已核实跨场景不同自然人。当前 13 条主线大多数仍是站立/上肢小动作，持续行走覆盖尚不足；不会为凑动作类别改标签。',
 '', '## 二、现在实际执行的完整链路','',
 '```text\nWaymo 原图、同一人物轨迹、时间戳\n  → 实际时间窗与原生像素审计\n  → 裁剪、RIFE 插值、SAM2 指定人物分割\n  → GVHMR 运动恢复与选人/骨架/构图检查\n  → 官方 4DAnyone-Turbo 公开权重：24 路同步动态视频\n  → 各生成视角分割、RGBA、相机与时间导出\n  → 21 训练视角/3 留出视角；训练视角轮廓交集初始化\n  → 独立 FreeTimeGsVanilla 逐资产优化\n  → checkpoint、.ftgs.ply、完整动作视频、固定时刻环绕\n  → 技术验收、生成一致性指标、视觉复核\n```',
 '', '前端只做公开权重推理，没有重新训练或微调 4DAnyone。后端“训练”指每条人物轨迹的动态高斯优化。当前后端是独立 FreeTimeGsVanilla，不是 4DAnyone 作者未发布的 FreeTimeGS 实现；原仓库的静态 Nerfstudio 教程没有用于冒充动态结果。[官方说明](https://github.com/ant-research/4DAnyone/blob/main/docs/nerfstudio.md)。',
 '', '现有主分支通常是 49 张真实约 10 Hz 观测，映射为 121 个 25 Hz 时间槽；其中 25 个槽直接保留实拍帧、96 个槽来自插值。121 不是 121 张独立实拍，也不是所有短轨迹都必须有 121 张照片。每视角视频为动态序列，不是单张静态图片。',
 '', 'walk063 在生成前分离根节点水平位移，保留垂直起伏、关节动作和转身，再保存可逆的逐帧变换。已经重施加原轨迹和一条明确标为合成的新轨迹；尚未实现任意 SMPL-X 动作替换，也未完成 Waymo 道路世界坐标/米制尺度配准。',
 '', '## 三、GPU 4–7 现在在做什么','',
 '|物理 GPU|任务|实际点数|当前步数 / 50,000|进度|PID|','|---|---|---:|---:|---:|---:|']
 for j in snapshot['jobs']:
  label=j['name'].removesuffix('_budget_half_50k').removesuffix('_capacity50k');dense=j['name'].endswith('capacity50k')
  count=read('capacity/'+label+'/manifest.json')['points'] if dense else (17056 if label.startswith('walk063') else 14939)
  name=('持续行走 walk063' if label.startswith('walk063') else '迈步/转身候选 0084')+('，增加容量' if dense else '，原容量')
  lines.append(f'|{j["gpu"]}|{name}|{count:,}|{j["step"]:,}|{j["step"]/500:.1f}%|{j["pid"]}|')
 lines+=['','四个训练进程均由实际 PID 和命令核验存活。固定同一输入/掩码/相机/时间划分、50k 计划、损失和 seed；增加容量组仅替换初始化点集。新点来自更细的轮廓交集表面，不是简单复制点或只提高未使用的上限。初始化几何、采样密度和近邻尺度也会改变，因此准确称为“容量/初始化密度对照”。',
 '', '后台会话：`tmux -L fdwaymo` 下的 `round2_mobility` 负责训练，`round2_mobility_after` 等待训练成功后接续整图三项指标与同步对照视频；失败会保留输出并停止相关接续，不静默降低质量。断开 IDE 不影响 tmux；服务器重启仍需依据检查点显式恢复。',
 '', '根据本次日志剩余时间，训练本体约再需 11–14 分钟；连同多检查点评测及视频渲染，保守估计 **15–25 分钟**。这是本次快照估算，共享任务和 I/O 会影响耗时。',
 '', '本次 GPU 利用率为 20%/20%/20%/23%，本项目训练每卡约 0.7 GiB 显存。四卡都有任务，但没有满载；低显存和低利用率不能当成训练失败，也不能声称已达到最佳吞吐。后续用实测区分 CPU 数据读取、Python 调度、CUDA 小算子等瓶颈，保持当前控制实验不变。',
 '', f'磁盘剩余约 {snapshot["free_disk_gib"]:.1f} GiB，当前四项可继续；扩大生成前重估空间，不直接承诺 100 条规模。允许 GPU 4–7 与其他项目共享；不使用 0–3，不终止其他项目。',
 '', '## 四、目前实验说明了什么','',
 '**1. 单纯增加训练步数有收益，但已出现收益递减。** 两条调参轨迹从旧 5k 到独立 15k 计划，灰背景前景 PSNR 提升约 0.712/1.090 dB，裁剪 LPIPS 均改善。再从 15k 到 50k，PSNR 仅再提高约 0.352/0.396 dB；第二条 LPIPS 还略变差，前景外 alpha 两条均略增加。更长训练没有自行消除模糊/漂浮点。[预算结果](budget_results.json)、[50k 结果](long_budget_results.json)。',
 '', '**2. 原生分辨率不是目前可以直接推广的单项改进。** 同样 15k，在统一原生网格比较时，704×1280 监督相对 352×640 监督的前景 PSNR 提升约 0.455/0.441 dB，但两条裁剪 LPIPS 和外部 alpha 都变差。不能仅凭 PSNR 提升宣布全面更好。[原生对照](native_results.json)。',
 '', '**3. 视角更多不等于固定少量步数下效果一定更好。** 共同初始化、同为 5k 时，两条均是 12 个训练视角优于 6 和 21。每个“视角×时刻”的更新次数随视角增加而减少，这是重要混杂因素；不能得出“12 视角普遍最优”。[视角对照](quality_results.json)。',
 '', '**4. 增加真实有效容量是当前证据较明确的方向。** 下表为整图、白背景、全 121 帧、相同 50k 预算对照，与前面灰背景前景/裁剪指标口径不同：','',
 '|轨迹|角色|ΔPSNR dB ↑|ΔSSIM ↑|LPIPS 相对下降|','|---|---|---:|---:|---:|']
 for p in bench['paired_results']:
  d=p['q50_to_dense50'];lines.append(f'|{p["label"]}|'+('调参' if p['role']=='tuning' else '独立验证')+f'|{d["full_psnr"]:+.3f}|{d["ssim"]:+.4f}|{p["lpips_relative_reduction_pct"]:.2f}%|')
 lines+=['','独立样本抽样图中，脸部轮廓、衣服边缘、手臂/手机比 5k 清楚；但包和头发的条纹状漂浮点、手部模糊仍存在，不能仅凭指标批准最终质量。[独立样本复核](reviews/independent_density.json)。当前正在训练的四组正是检验此配方在行走和迈步/转身上的适用性。',
 '', '**5. 移动相机真实标定接入成功，并不等于人体拟合一定更好。** R01 已验证原始相机参数、坐标轴、曝光时刻、畸变、逐帧裁剪后的 K 与相机旋转，并完成相同输入/特征的静态估计和移动标定两组推理。相对检测关键点，中位重投影误差由 5.46 px 变为 7.02 px，没有显示改善。检测点不是真值；两组的肢体投影都大致对齐，但遮挡、口袋内手部和脚部仍需复核。暂不提升标定分支为默认。[诊断视频](experiments/moving_R01_v1/motion_pair/projection_comparison.mp4)、[数值](experiments/moving_R01_v1/motion_pair/comparison.json)、[复核](experiments/moving_R01_v1/motion_pair/review.json)。',
 '', '## 五、评测范围与质量缺口','',
 '8 条基线每项检查 3 个留出生成视角 × 121 帧＝363 对图像。指标为整图 PSNR、SSIM、AlexNet LPIPS，另保留前景误差、轮廓与外部 alpha。已有指标交叉核验和相同图像的 SSIM/LPIPS 自检；配对检查参考图和配置的哈希。[完整表](benchmark/results.csv)、[协议](benchmark/protocol.json)、[验证](benchmark/verification.json)。',
 '', '按照最新要求，缺少真实新视角真值时，不计算“生成对真实新视角”“高斯对真实新视角”的伪指标，也不为凑论文表格去复现不同数据集/相机条件。当前分数只能衡量对生成观测的拟合/一致性，不能证明背面、脸部或附件符合真实人物。',
 '', '已保留的明确问题：0054 的脸颈生成异常；0062 的头发/身份外观变化；0055 栅栏遮挡；0084 初段遮挡；多条包、链带、手与脚的细节损失。生成阶段已经错的内容，增加高斯训练可能只会把错误拟合得更准确。',
 '', '行走 walk063 已具备局部动作和轨迹回放，但尚无可靠脚接地、世界尺度、道路遮挡、任意动作重定向验收。其 5k 留出一致性基线较差，是当前优先检查的动态样本。',
 '', '## 六、后续工作，按优先级执行','',
 '|优先级/状态|具体工作|要回答的问题|交付/继续条件|','|---|---|---|---|',
 '|P0 已在运行|完成当前四组 50k，自动评测与视频对照|密度配方能否改善行走和转身，而非只对站立有效？|两条各自的 5k/原50k/密50k 同相机视频与指标；失败保留|',
 '|P1 接续当前结果|检查首中末及异常时段的手脚、包、拖影和外部 alpha|指标收益是否同时带来动作细节改善？|逐条通过/失败判定，不只看均值|',
 '|P1 条件实验|若密度增加仍留下明显动态拖影，检查时间中心/持续时间/速度与实际采样覆盖|主要限制是否来自时间表示或不一致监督？|先诊断，再做一个变量的小对照；不直接增加到100k或全因子扫参|',
 '|P2 已有诊断，未生成|移动相机 R01 做指定目标分割、遮挡审计；必要时改用更清楚的已审计候选|自车运动、移动裁剪和人体运动是否可靠区分？|投影、身份、构图通过后才做六视角试验；不把未过检分支提交大批量|',
 '|P2 待专项排查|量测加载/CPU/CUDA 各阶段耗时，优化低利用率|四卡为何有任务但未满载？|在保持数值及采样定义可比的条件下提高吞吐；另记性能变化|',
 '|P3 待执行|附件掩码、短观察片段分别做小实验|包/链带丢失是否来自输入；短轨迹如何保留真实时长？|短分支已有21帧与固定latent长度冲突诊断，尚未修复；不伪造长实拍序列|',
 '|后续接入阶段|度量尺度、世界坐标、地面、场景遮挡与资产接口|能否真正插入驾驶场景？|当前未完成；轨迹平移演示不能替代场景融合/动作重定向|',
 '', '先完成当前小批量实验与逐阶段结论，再决定是否扩大到 24 条；100 条保持后续候选，不将未验证配方一次提交。此前建议的“更多行走/转身”仍是样本覆盖目标，当前覆盖不足会如实保留。',
 '', '## 七、可以直接查看的文件','',
 '- [全部 demo 页](gallery/index.html)：旧 12 条、新行走和保留失败。',
 '- [基线指标及四条配对](benchmark/REPORT.md)：8 条的 PSNR/SSIM/LPIPS 与源视频链接。',
 '- [行走四联屏](reconstruction/outputs/walk063_rootlocal24/demo/four_panel.mp4)、[固定相机动作](reconstruction/outputs/walk063_rootlocal24/demo/fixed_views.mp4)、[新轨迹回放](reconstruction/outputs/walk063_rootlocal24/demo/replay_new.mp4)。',
 '- [行走动态 PLY](reconstruction/outputs/walk063_rootlocal24/ckpt_4999.ftgs.ply)、[资产坐标/时间说明](reconstruction/outputs/walk063_rootlocal24/asset.json)、[手脚与附件细节](reconstruction/outputs/walk063_rootlocal24/demo/details/view00_frame060.jpg)。',
 '- [0073 优化对照视频](comparisons/batch_0073_twentyfour_validation/view08.mp4)、[0086 优化对照视频](comparisons/batch_0086_twentyfour_validation/view08.mp4)。',
 '- [后续自动评测/渲染队列](mobility_after.json)。新行走优化产物只有在实际完成后才会作为结果链接加入报告。',
 '', '## 八、版本与记录','',
 '项目 `/mnt/ojc/workplace1/4danyone`，新实验 `waymo/round2/`。复用专用 `4danyone`、`4danyone-sam2`、`4danyone-ftgs` 环境；未改变旧模型环境或公共数据。官方源码固定提交 `9cc2aa230fe5d364da2c5634ec3dabadadcb0d60`，GVHMR `6ec3ca39336c50492c0fae65fba2fb831fc7d866`。独立后端和本地修改见 [源码补丁](quality_trainer.patch)、[代码哈希](code_manifest.json)。',
 '', '[阶段状态](status.json)、[实验矩阵](experiment_matrix.json)、[样本审计清单](sample_manifest.json)、[本次四组配置](mobility_plan.json)、[GPU共享规则](gpu_sharing_policy.json)。所有未通过和历史失败记录保留；后台运行不等于质量验收。']
 p=R/'REPORT.md';archive=R/'report_snapshots';archive.mkdir(exist_ok=True)
 name='before_progress_'+stamp.replace(':','').replace('+','_')+'.md'
 if p.exists():(archive/name).write_bytes(p.read_bytes())
 p.write_text('\n'.join(lines)+'\n')
 print(str(p))
if __name__=='__main__':main()
