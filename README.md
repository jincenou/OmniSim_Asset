# OmniSim Asset：刚体与人体资产构建代码

本仓库整理 **驾驶日志／网络输入 → 刚体 3D Gaussian 或人体 4D Gaussian → 坐标与尺度规范化 → 质检、导出及资产库登记** 的实际业务代码，并按阶段索引保留过往构建实验。源码快照日期为 **2026-11-22**。

这里发布源码、配置模板和来源记录；资产库本体、模型权重、数据集、训练输出、相机／动作张量及实时队列保留在运行服务器。历史实验代码保留原算法和文件关系，不把已有结果写成这次重新训练或全链路复现的结果。

## 从哪里开始

| 需求 | 入口 |
|---|---|
| 统一刚体构建，nuScenes／Waymo／图片帧／视频 | [`source/DriveLab/asset_extraction/run.py`](source/DriveLab/asset_extraction/run.py) |
| 便于配置部署的人体构建 | [`source/DriveLab/human/run.py`](source/DriveLab/human/run.py) |
| 原人体生产批次与其局部更新 | [`source/workplace1/human/run.py`](source/workplace1/human/run.py) |
| 刚体逐场景生产、类别轮转与去重 | [`source/workplace1/asset_work/scene_production_20261031/`](source/workplace1/asset_work/scene_production_20261031/) |
| 网络视频人体接入 | [`source/workplace1/4danyone/web_people/`](source/workplace1/4danyone/web_people/) |
| 刚体／人体流程及输入输出 | [docs/PIPELINES.md](docs/PIPELINES.md) |
| 依赖、环境和运行顺序 | [docs/RUN.md](docs/RUN.md) |
| 历史实验归纳 | [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) |
| 全部源码索引和 SHA256 | [provenance/source_manifest.json](provenance/source_manifest.json) |

跨模型定量评测分别在 [human_model_comparison](https://github.com/jincenou/human_model_comparison) 与 [vehicle_model_comparison](https://github.com/jincenou/vehicle_model_comparison)。本仓库中的构建阶段诊断与生成视角验证，不能替代这些仓库的真实观测评测。

## 目录与可运行范围

```text
source/
  DriveLab/                    # 统一刚体与人体入口，便于配置的部署版本
  pre_flux_pipeline/           # 早期刚体前端，历史依赖
  workplace1/
    human/                    # 原人体生产业务代码和测试
    asset_work/               # 不同来源与批次的生产控制器
    4danyone/                  # 视频前端、后端配方与网络人体实验
    reconviagen_demo/scripts/  # 刚体重建、视图数/朝向对照、规范化
    human_cycle_update_20261108/
    human_minimal_toolkit_20261105/
    emergency_sign_asset/scripts/
    asset_library/maintenance/ # 维护代码；不包含资产数据
docs/                         # 流程、依赖、实验索引
tools/                        # 只读校验和隔离工作区物化
tests/                        # 路径迁移与文件保护测试
provenance/                   # 来源、依赖版本与文件哈希
```

`source/` 是保留原布局的源码快照，不能直接在其中启动历史生产任务。部分旧控制器依赖当时的候选清单、机器路径或检查点；这些历史入口在实验索引中单列。优先使用 `DriveLab` 的配置入口，复现实验时再查相应快照。

## CPU 检查与工作区准备

以下命令只检查源码或复制代码，不调用模型、不下载数据、不启动训练：

```bash
python tools/verify_source.py
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v

# 默认只打印计划。路径使用不含空格的绝对路径。
python tools/materialize_workspace.py \
  --workspace-root /data/omnisim_workspace \
  --envs-root /opt/conda/envs \
  --datasets-root /datasets

# 在另一个目录创建可配置的源码工作区；拒绝覆盖任何已有文件。
python tools/materialize_workspace.py \
  --workspace-root /data/omnisim_workspace \
  --envs-root /opt/conda/envs \
  --datasets-root /datasets --apply
```

工具只将已知绝对路径前缀映射到明确指定的位置，生成逐文件收据；源快照及原资产库均不改动。额外机器路径可用 `--path-map /path/to/paths.local.json` 显式指定。模型源码、数据、权重、批次登记及 CUDA 环境仍需按 [运行说明](docs/RUN.md) 配置。

三仓库共用的实际布局：OmniSim 的 `--workspace-root` 是 `<ROOT>`；车辆仓库物化时使用 `--workspace <ROOT>/workplace1`；人体评测仓库设置 `HMC_WORKSPACE=<ROOT>/workplace1`。两类评测目录与资产流水线均位于同一个 `workplace1` 下。

## 来源与文件保护

所有收录文件均记录原相对路径和 SHA256；未收录资产、媒体、权重、环境安装目录、第三方仓库副本及逐资产运行记录。第三方固定来源及恢复方法见 [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md)，原声明保留，不为整包追加覆盖第三方的统一许可证。

资产库及其外部引用的真实 payload 必须按 [docs/ASSET_PROTECTION.md](docs/ASSET_PROTECTION.md) 保护。这里整理的是代码，**没有执行资产清理或重写已有结果**。
