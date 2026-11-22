"""Generate final report from evaluated, replayed, and visually reviewed artifacts."""
import json,datetime,os,hashlib
from pathlib import Path
from data import ROOT,save_json,sha
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
objects={};rows=[];cost=[];confrows=[]
for label,oid in [('R07','OBJ_0018'),('M02','OBJ_0019')]:
 clip=ROOT/'objects'/oid/'clips/C001';d=json.loads((clip/'08_evaluation/comparison.json').read_text());objects[label]=d
 for variant in ['E0','E1']:
  x=d['runs'][variant];m=x['metrics']['all61'];rows.append(f"|{label} {variant}|{m['gray_foreground_psnr']:.3f}|{m['gray_crop_alex_lpips']:.4f}|{m['silhouette_iou']:.4f}|{m['outside_alpha_mean']:.5f}|{m['lower35pct_recall_alpha05']*100:.2f}%|")
  cost.append(f"|{label} {variant}|{x['points']}|{x['steps']}|{x['rendered_samples']}|{x['training_seconds']:.1f}s|{x['peak_allocated_bytes']/2**20:.1f}MiB|")
 c=json.loads((clip/'08_evaluation/confidence_regions.json').read_text())['summary'];confrows.append(f"|{label}|{c['foreground_weight']:.3f}|{c['lower35pct_weight']:.3f}|{c['high_temporal_change_weight']:.3f}|{c['low_temporal_change_weight']:.3f}|")
links='[R07 / OBJ_0018](objects/OBJ_0018/clips/C001/index.html) · [M02 / OBJ_0019](objects/OBJ_0019/clips/C001/index.html) · [独立总入口](gallery/index.html)'
text=f'''# Efficient4D 原生4D后端独立接入实验

更新：{now}（北京时间）。{links}

已实际完成两对象的E0/E1训练、完整动作与环绕、独立动态重载、61时刻×3留出视角评价、置信度进入loss/gradient验证。**技术链路完成不等于质量改善；生产视觉批准0。** 本轮为复旦公开generation分支的本地重建适配，不是Efficient4D整篇论文复现。

## 输入、采样与预算

- 已有24视角×61时间槽，352×640 straight RGBA，25fps，首末跨度2.4秒（编码时长2.44秒）。来源25张实拍；13槽直接保留、48槽插值。未重训4DAnyone、未重跑SyncDreamer、未补帧伪造长度。
- 正式训练视角：1,3,5,7,9,11,13,15,17,19,21,23，按实际方位每30°选取。验证相机0/8/16。31个训练时刻为帧0,2,…,60（间隔0.08秒）；输出和评价0…60，奇数帧是未直接监督的时间查询。
- 每步microbatch1，累积4次渲染；12,000 optimizer steps、48,000 rendered samples。每个训练观测被采样129或130次，精确覆盖见各complete.json。初始化与采样RNG独立；E0/E1初始参数哈希、完整采样序列哈希和逐观测计数均相同。
- 新建训练视角轮廓表面，7个允许时刻0/10/20/30/40/50/60；无验证视角、无未监督时间泄漏。固定点数，不增密、不剪枝。空间sigma0.022、时间sigma0.15模型单位（0.36秒）是明确的本地配方，非照搬FTGS sigma8。
- E0为RGB MSE＋全图alpha MSE；E1仅RGB由真实RIFE置信度加权，两者按像素权重和归一化。alpha始终全图、不按confidence屏蔽。无额外扩散权重加载。

## 可配置接口验证

`configure_run.py --help`可独立指定训练相机、训练时刻、初始化时刻、真实时间窗、输出fps和梯度累积。配置写入新manifest，原实验清单不覆盖；改变真实窗口必须新clip目录。实测7视角×7时刻、0.4–1.6秒、batch累积2，20步训练与独立重载通过，31帧25fps完整输出。不等距RIFE邻域比例也实测通过；泄漏和伪造长度被拒绝。证据：research/interface_test/result.json。这些是接口测试，不计入四组正式质量实验。

## 配对评价结果

参考为留出**生成观测**，前端可能跨视角交换信息，且这两对象已经用于调参；不是独立最终测试，也不是真实Waymo新视角真值。以下为3视角×61时刻均值，灰背景；置信度不参与评价筛选。

|对象/运行|前景PSNR↑|裁剪Alex LPIPS↓|IoU↑|mask外平均alpha↓|底部35%覆盖↑|
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

白背景整图PSNR/SSIM/Alex LPIPS、灰背景前景/裁剪指标分栏保存在各`evaluation/metrics.json`；另分监督时刻、奇数插值时刻与旧13时刻子集。底部35%是目标mask外接框下部代理，不是真实脚部标注精度。

'''
for label,d in objects.items():
 e0=d['runs']['E0']['metrics']['all61'];e1=d['runs']['E1']['metrics']['all61'];delta=(e1['gray_crop_alex_lpips']/e0['gray_crop_alex_lpips']-1)*100
 text+=f"{label}：E1−E0 前景PSNR {e1['gray_foreground_psnr']-e0['gray_foreground_psnr']:+.4f}dB，裁剪LPIPS {delta:+.2f}%（正值为变差），IoU {e1['silhouette_iou']-e0['silhouette_iou']:+.5f}。单seed描述性结果，不作统计显著性结论。\n\n"
text+='''## 旧后端参考与视觉结论

B0为R07/M02旧sigma8、15k优化结果，使用21×61观测；本轮12×31且48k图像更新、损失/初始化/动态表示与SH容量不同（本轮degree0，旧B0 degree3）。因此B0只是参考，不能形成纯后端优劣排名。对照视频显著标注不匹配预算。B0视频从已验收旧编码结果读取，定量指标来自原始旧评测，未从有损视频伪算指标。E0/E1使用原生未滤点结果；没有以滤点隐藏缺陷。

'''
for label,d in objects.items():
 old=d['B0']['legacy_metrics'];text+=f"{label}旧B0（3×13时刻、灰背景）前景PSNR {old['foreground_psnr']:.3f}，裁剪LPIPS {old['crop_lpips']:.4f}，IoU {old['silhouette_iou']:.4f}。对应新后端13时刻子集见comparison.json；全61均值不与旧13均值混排。\n\n"
for label,oid in [('R07','OBJ_0018'),('M02','OBJ_0019')]:
 review=json.loads((ROOT/'objects'/oid/'clips/C001/08_evaluation/visual_review.json').read_text());text+=f"**{label}抽样视觉复核：** "+review['finding']+'\n\n'
text+='''R07条件协方差诊断发现中时刻活跃高斯长短轴比99分位约423，最大约1323；所有条件协方差特征值为正。作者Python协方差与CUDA路径的平均RGB差约1e-8；端点个别像素最大RGB差0.00139，其余查询约几e-6，不能宣称逐像素完全一致。该证据支持排查细长高斯形状退化，不证明单一因果。

## 置信度机制与E2决定

置信度基于本次生成RGBA灰背景合成图，同训练视角邻近±1/±2训练时刻，经RIFE预测当前帧，计算RGB/作者SSIM不一致。采用独立复制的Practical-RIFE4.25_0919，区别于作者旧HDv3权重。端点等权、常量误差等可靠性、零分母、空mask、非有限值和detach均有测试。真实训练中也保存了等权/加权loss及高斯颜色梯度变化，证明E1并非仅输出热力图。

|对象|前景平均权重|底部35%平均权重|高时间变化四分位|低时间变化四分位|
|---|---:|---:|---:|---:|
'''+chr(10).join(confrows)+'''

运动变化大的区域和下肢代理明显更易被降权。这可能包含真实动作、遮挡或生成不一致，不能据此认定全部都是错误观测；全图alpha约束保留。E1未展示足以解决当前缺陷的收益。

**本轮不执行E2。** 主路径若做生成指导，应命名Zero123.refine＋图像MSE，不能称已运行论文式SDS。当前生成参考已有清晰衣服/包细节，而重建存在明显细长高斯退化；增加生成先验会引入新因素，尚非最值得做的受控实验。下一项建议：保持S1观察、seed、预算、点数和损失不变，只加入条件空间协方差轴比正则，保留未正则E0作对照；先R07有界验证，若有效再M02。没有据本轮结果恢复100条资产生产。

## 验证、资产和资源

真实CUDA/KNN、矩形及偏心主点投影、时间/空间尺度/双4D旋转等梯度与更新均通过。真实相机数值投影最大误差0.000492像素；它只证明坐标转换，不证明人体真值。合成样例多时间查询差异非零；各正式checkpoint在新进程对3相机×4查询时刻重载，RGB/alpha最大误差0。

每对象`06_assets/E0`、`E1`保存完整原生checkpoint、模型构造参数、配置、manifest、根变换和重载证据。checkpoint保留xyz/时间/空间与时间尺度/双旋转/opacity/SH，以及optimizer/RNG/采样状态；没有用普通PLY冒充动态资产。`render_asset.py --help`是无需目标图像的独立渲染入口。root-local局部动作重放可用；真实道路世界配准、脚接地、任意换动作均未验收。

|运行|高斯点数|优化步数|训练渲染数|GPU训练阶段墙钟|PyTorch峰值分配|
|---|---:|---:|---:|---:|---:|
'''+chr(10).join(cost)+'''

峰值为PyTorch allocator记录，不含全部驱动上下文；时间是明确GPU进程的训练阶段墙钟，不是利用率积分。RIFE预处理和评价耗时有进程计时；CPU导入/环境/打包历史耗时未单独计时，research/costs.json记录脚本创建到退出的明确上界，不冒充精确计时。R07 E0/E1在GPU4/5，M02 E0/E1在GPU6/7；每个GPU进程核对实际CUDA UUID。允许共享，准入只代表当时剩余显存；不把空闲当永久预约。未使用GPU0–3，未终止其他任务。

新环境`/mnt/ojc/miniconda3/envs/eff4d`；源码/缓存/扩展/配置/输出均在NEW，长期任务用`tmux -L eff4d`。所有输入导入前后与训练前后hash核验，旧B0展示视频打包前后hash核验。没有写旧实验目录；不声称并行旧任务期间整个旧工程完全不变。没有公网服务或上传；publish_manifest只用于未来显式合并。

失败保留：初始沙箱看不到GPU（宿主只读检查后解决）、继承Git代理失效（仅单命令禁用）、网络conda元数据等待（仅停止本项目进程，改用校验归档）、首轮ninja位置错误（独立复制可执行文件后修复）、首次视频编码找不到ffmpeg（改用本环境包内二进制后全帧解码通过）。正式E0/E1未OOM、未改变预算或静默降级。质量失败与负结果如上，仍未生产批准。

[源码核查](SOURCE_AUDIT.md) · [冻结计划](PLAN.json) · [状态](STATUS.json) · [交接](HANDOFF.md) · [逐项交付审计](research/completion_audit.json)
'''
(ROOT/'REPORT.md').write_text(text)
(ROOT/'HANDOFF.md').write_text(f'''# Efficient4D独立实验交接

{now}

{links}

本轮R07/M02 E0/E1均实际完成12k训练、48k渲染采样、新进程重载、61×3评价和六类视频。详见REPORT.md及research/completion_audit.json。生产批准0，E1没有解决行走质量，细长高斯条纹需后续受控处理。E2未执行，原因见报告。勿覆盖v1或把未做E2说成成功。

实验目录与环境独立，旧waymo/round2和objects只读。授权GPU4–7可共享；0–3禁止。每次启动须实时检查剩余显存/磁盘，核对UUID，至少保留35GiB。长期任务使用tmux -L eff4d。旧任务的状态不归本对话管理。

恢复查看结果无需训练。新实验需新run_id，冻结变化；建议仅测试条件空间协方差轴比正则，其他S1因素固定。本轮没有执行此建议，也未实现任意动作重定向、道路世界配准或生产批准。

运行环境：source scheduler/env.sh；CUDA_VISIBLE_DEVICES=4 python adapters/render_asset.py --asset objects/OBJ_0018/clips/C001/06_assets/E1 --output objects/OBJ_0018/clips/C001/07_demo/new_replay.mp4 。所有输出必须位于eff4d真实路径。

训练支持保存optimizer/RNG和采样状态，但当前CLI未提供严格resume入口；不要把load后新计划当resume。需要恢复时应核验输入hash并明确实现恢复全状态，或标为新finetune。现有任务已自然完成，交接不要求重复训练。
''')
(ROOT/'gallery/index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Efficient4D 独立实验</title><style>body{font:18px system-ui;max-width:1000px;margin:60px auto;background:#111827;color:#e5e7eb}a{color:#7dd3fc}</style><h1>原生4D后端 · R07 / M02</h1><p>两对象E0/E1已训练、动态重载、评价并生成完整视频。12视角×31训练时刻，61帧输出、2.4秒跨度。B0为不同观测预算参考。生产视觉批准：0。</p><p><a href="../objects/OBJ_0018/clips/C001/index.html">R07 · 提袋行走</a></p><p><a href="../objects/OBJ_0019/clips/C001/index.html">M02 · 肩包行走／移动自车</a></p><p><a href="../REPORT.md">完整报告与负结果</a></p>''')
save_json(ROOT/'publish_manifest.json',{'schema':1,'published':False,'scope':'Future explicit merge only; no old gallery modified','objects':[{'id':oid,'entry':str(ROOT/'objects'/oid/'clips/C001/index.html')} for oid in ['OBJ_0018','OBJ_0019']],'report':str(ROOT/'REPORT.md')})
# Adapter source fingerprints; captured after runs (not represented as at-launch hashes).
paths=list((ROOT/'adapters').glob('*.py'))+list((ROOT/'scheduler').glob('*.sh'))
save_json(ROOT/'research/code_manifest.json',{'captured_at':now,'scope':'final source snapshot; not a claim of at-launch collection','files':{str(p.relative_to(ROOT)):sha(p) for p in paths}})
print(ROOT/'REPORT.md',flush=True)
