# 实验与代码索引

这里按实验目的归纳实际源码，保留独立阶段及其原目录；文件数量不是完成样本数。历史 GPU 作业脚本和批次状态不能直接作为新运行配置。跨模型最终评测请使用两个 comparison 仓库。

| 模块 | 源码范围 | 文件数 | 用途与边界 |
|---|---|---:|---|
| 统一刚体与人体入口 | [`DriveLab`](../source/DriveLab/) | 95 | 配置化执行、来源接入、生成、导出与质检；作为新部署入口。 |
| 原人体生产实现 | [`workplace1/human`](../source/workplace1/human/) | 36 | 保留生产队列、固定三阶段优化、实际资产导出与后续局部更新。 |
| 早期前端 | [`pre_flux_pipeline`](../source/pre_flux_pipeline/) | 24 | 早期刚体前端快照；保留部分历史批次的导入关系。 |
| 刚体逐场景批次 | [`workplace1/asset_work/scene_production_20261031`](../source/workplace1/asset_work/scene_production_20261031/) | 37 | nuScenes/Waymo 逐场景处理、类别优先级、身份去重和资源租约。 |
| 车辆数量扩展 | [`workplace1/asset_work/vehicle1000_20261025`](../source/workplace1/asset_work/vehicle1000_20261025/) | 4 | 历史车辆批量构建与身份检查；固定目标只属于当时批次。 |
| 首轮刚体输入与批处理 | [`workplace1/asset_work/batch20_20261024`](../source/workplace1/asset_work/batch20_20261024/) | 19 | 驾驶日志输入适配、前端筛查和初始批次控制。 |
| 网络车辆与骑行者 | [`workplace1/asset_work`](../source/workplace1/asset_work/) | 181 | web_vehicles、openimages_*、vehicles*_json/recovery 等来源接入与补录；该行含同目录其他历史批次。 |
| 刚体重建／视图实验 | [`workplace1/reconviagen_demo/scripts`](../source/workplace1/reconviagen_demo/scripts/) | 263 | 真实视图数、FLUX 视图、朝向、非车辆、尺度、场景插入与对应诊断。job_*.sh 是历史任务参数。 |
| 早期官方模型部署 | [`workplace1/4danyone/scripts`](../source/workplace1/4danyone/scripts/) | 12 | 模型准备、运行检查和初始生成部署脚本。 |
| Waymo 输入与动作 | [`workplace1/4danyone/waymo/scripts`](../source/workplace1/4danyone/waymo/scripts/) | 26 | 真实轨迹裁剪、插帧、分割、动作与相机契约检查。 |
| 前端几何与遮挡实验 | [`workplace1/4danyone/waymo/frontend_general_v1`](../source/workplace1/4danyone/waymo/frontend_general_v1/) | 156 | 原始画幅动作、裁剪投影、骨架条件、遮挡修复及独立诊断；保留实验方法，不统一宣称采纳。 |
| 候选池筛查 | [`workplace1/4danyone/waymo/baseline_pool`](../source/workplace1/4danyone/waymo/baseline_pool/) | 94 | 身份关联、源边界检查、候选筛选和已筛选包准备。 |
| 早期后端与预算实验 | [`workplace1/4danyone/waymo/round2`](../source/workplace1/4danyone/waymo/round2/) | 261 | FTGS/4DGaussians 的时间、视角、预算、损失和导出对照；外部后端代码不重复收录。 |
| 后端横向研究 | [`workplace1/4danyone/waymo/backend_study`](../source/workplace1/4danyone/waymo/backend_study/) | 44 | 后端比较的适配、计划与评测脚本；不是最终人体基线的统一入口。 |
| Grid4D 配方与改进实验 | [`workplace1/4danyone/waymo/grid4d`](../source/workplace1/4danyone/waymo/grid4d/) | 300 | 基础适配、边界监督、quality_v3、cross8、query0_fused、EVQ 与 schedule_fusion；现行生产不自动启用全部变体。 |
| 随机轨迹／场景扩展 | [`workplace1/4danyone/waymo/random8`](../source/workplace1/4danyone/waymo/random8/) | 103 | 独立轨迹扩展及对应检查。 |
| 场景批量扩展 | [`workplace1/4danyone/waymo/scene20`](../source/workplace1/4danyone/waymo/scene20/) | 21 | 历史扩展队列与输入／输出检查。 |
| Efficient4D 对照 | [`workplace1/4danyone/eff4d`](../source/workplace1/4danyone/eff4d/) | 43 | E0 RGB+alpha、置信度 E1 等后端适配与调度；不作为当前 Grid4D 配方。 |
| 网络人体接入 | [`workplace1/4danyone/web_people`](../source/workplace1/4danyone/web_people/) | 97 | 2026-10-21 完成批次与 2026-11-16 后续输入实验分开保留，不合并状态或成功结论。 |
| 携带物处理 | [`workplace1/4danyone/preprocessing`](../source/workplace1/4danyone/preprocessing/) | 6 | 可见携带物掩码与原始像素恢复辅助代码。 |
| 连续资产最小化 | [`workplace1/human_minimal_toolkit_20261105`](../source/workplace1/human_minimal_toolkit_20261105/) | 1 | 独立导出、连续 runtime 与回放完整性检查。 |
| 循环回放／动作分类 | [`workplace1/human_cycle_update_20261108`](../source/workplace1/human_cycle_update_20261108/) | 31 | 原版本、v2、v3 和验证脚本；与原源时间序列评测区别记录。 |
| 应急标识 | [`workplace1/emergency_sign_asset/scripts`](../source/workplace1/emergency_sign_asset/scripts/) | 31 | 来源采集、分割、重建、组合、入库和插入示例。 |
| 库格式与骑行者维护 | [`workplace1/asset_library/maintenance`](../source/workplace1/asset_library/maintenance/) | 85 | 规范化、来源与加载检查、骑行者固定组合体处理；cleanup 入口不自动执行。 |

上表存在包含关系（例如 `asset_work` 总目录与其子批次），数量不能相加作为总数。完整收录清单以 [source_manifest.json](../provenance/source_manifest.json) 为准。

## 当前入口与历史入口

- **新部署：** 先配置 `DriveLab/asset_extraction/run.py` 或 `DriveLab/human/run.py`。
- **原实验回放：** 使用对应历史目录及其记录的输入／运行条件；不把所有 `job_*.sh` 一次性执行。
- **算法对照：** loss、view、budget、query 等变体仅代表被实现和试验的方案，未在此宣称全部具有收益或被最终采用。
- **导出和质检：** 技术可加载、源视角一致性、生成视角验证和实际视觉通过是不同记录。
- **数据与配置：** 逐资产样本清单、产物和多数自动生成的场景配置未发布；已保留协议模板与代码。

## 源码完整性

源文件保持原始字节，SHA256 可校验。没有把业务代码的未提交改动丢弃为远端旧版本；对应源 commit 仅用于描述依赖出处。源码整理本身不重算任何论文指标。
