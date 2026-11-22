# 构建流程与代码对应

## 刚体

```text
nuScenes / Waymo / 网络图片 / 网络视频
→ 源帧、身份与相机信息
→ Grounding DINO 检测与关联、SAM2 前景分割
→ 输入质量筛查、Qwen3-VL 属性描述
→ 真实互补视图选择 + FLUX 参考视图生成
→ 车辆 ReconViaGen / 其他刚体 TRELLIS
→ 朝向、接地和等比例尺度规范化
→ 环绕渲染与质量记录
→ 导出 Gaussian、来源记录和资产库清单
```

以下路径均相对 `source/DriveLab/`。

| 阶段 | 实现 | 输入 → 输出 |
|---|---|---|
| 数据源适配 | `asset_extraction/sources.py`、`nuscenes/prepare_routes.py` | 数据目录／视频／帧目录 → 源帧、标定与路由 |
| 检测与分割 | `asset_extraction/pipeline.py`、`vision.py` | 真实观测 → 目标轨迹、mask、前景候选 |
| 输入筛查与属性 | `asset_extraction/vlm.py`、`nuscenes/quality_filter.py` | 原始候选 → 输入记录、可见属性与筛查结论 |
| 多视图选择 | `asset_extraction/real_views.py` | 同一对象观测 → 真实参考视图及相机信息 |
| 生成式补全 | `asset_extraction/flux.py`、`generation.py` | 前景与属性 → 补全参考视图及模型任务 |
| 3D 重建 | `asset_extraction/generation.py` | 真实／生成视图 → ReconViaGen 或 TRELLIS 输出 |
| 规范化与质检 | `asset_extraction/gaussians.py`、`export.py`、`quality.py` | 重建结果 → 规范化资产、渲染诊断和质量记录 |
| 发布与去重 | `asset_extraction/rigid_publication.py`、`bank.py` | 已记录结果 → 可追溯的库条目 |
| 统一调度 | `asset_extraction/run.py` | pipeline YAML → 分阶段执行、断点收据 |

`source/workplace1/asset_work/scene_production_20261031/` 保留实际逐场景生产控制器：`selection_scope.py` 固定选择范围，`identity.py` 管理身份，`vehicle_priority.py` 与 `run.py` 管理类别顺序，`runtime.py` 负责资源租约与入库。它依赖历史批次登记，不应直接套用旧状态文件生成新批次。

`source/workplace1/reconviagen_demo/scripts/` 保留车辆视图数、朝向、真实视图与生成视图组合、非车辆以及插入场景实验。具体阶段脚本按实验索引检索，不把各个 `job_*.sh` 当成当前统一入口。

## 人体

```text
可用的真实人物轨迹与时间戳
→ 源帧裁剪、RIFE 插值和前景分割
→ GVHMR 姿态恢复、水平根运动分离
→ 源外观 + 局部运动 + 目标相机和投影骨架
→ 4DAnyone Turbo 24 路同步视频
→ 软 RGBA、21 训练视角 / 3 生成参考留出视角
→ visual hull 初始化 + Grid4D 三阶段优化
→ 资产导出、独立回放、局部动作和场景轨迹解耦
```

以下核心模块在 `source/DriveLab/human/` 与 `source/workplace1/human/` 均保留。前者是集成配置入口；后者保留原生产批次代码，包括后续局部队列更新。不要混拷两个版本的单个模块。

| 阶段 | 实现 | 核心契约 |
|---|---|---|
| nuScenes 输入 | `intake.py`、`intake_motion.py` | 保留真实曝光时间、裁剪 K、身份与估计动作；插值不增加真实观测 |
| 已筛选 Waymo 输入 | `batch.py`、`prepare.py` | 候选 CSV / G0 包必须具备视频、动作、相机、骨架和 root motion |
| 外观和动作条件 | `prepare.py`、`frontend.py` | 可见携带物、原始像素、局部动作及 24 路相机条件一致 |
| 多视角生成 | `frontend.py` | 冻结 Turbo 生成，输出同步视频和相机记录 |
| RGBA 与初始化 | `dataset.py`、`data.py` | 四组独立分割，训练视角初始化，留出视角隔离 |
| 时空重建 | `reconstruction.py` | 32,999 基础更新 + 15,000 轮廓监督 + 5,000 低学习率细化 |
| 生成参考诊断 | `evaluation.py` | 生成视频上的一致性诊断，不是真实新视角真值 |
| 导出 | `assets.py` | 完整 SH、协方差、时间、相机与来源记录；独立重载验证 |
| 场景轨迹控制 | `placement.py`、`playback.py` | 局部动作与水平根轨迹分开，尺度及坐标变换只应用一次 |
| 队列与增量检查 | `batch.py`、`audit.py` | 已登记身份、对象锁、断点状态和实际质量复核 |

人体连续 runtime 最小化在 `source/workplace1/human_minimal_toolkit_20261105/compact_humans.py`。循环回放相关版本在 `human_cycle_update_20261108/`，它们是独立后处理实验；不能将任意源片段都解释成经过验证的完整动作周期。

## 网络视频人体接入

`source/workplace1/4danyone/web_people/r20261021_114925/scripts/` 保存已完成网络人体实验的接入与导出流程。`r20261116_sources_v1/` 保存后续接入实验及其隔离人体代码快照。

1. `source_io.py`、下载与检查脚本保留源视频及时间信息，`track_video.py` 生成身份轨迹。
2. `prepare_person.py` 处理裁剪、分割及动作输入，`motion_original_frame.py` 在原始画幅中估计动作，再转换裁剪 K 和投影关键点。
3. 后续 `build_conditions.py`、`register_g0.py` 组织相机、骨架及 G0 契约；2026-11-16 的入口不自动批准未审核的输入。
4. 由对应前端／后端阶段接入同一人体重建配方。近似静止相机、估计内参和尺度均须在来源记录中保留。

网络图像和视频不是额外的模型真值；本仓库保留接入实现，不复制原视频或资产，也不改变既有定量评测集合。
