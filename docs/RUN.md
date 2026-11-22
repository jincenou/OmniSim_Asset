# 部署与运行顺序

## 1. 检查源码，再配置执行树

仓库根目录运行 `python tools/verify_source.py`。它检查 source 清单、SHA256、Python AST、JSON 和 shell 语法，不导入模型。`python -m unittest discover -s tests -v` 检查物化工具不会覆盖已有文件或进入符号链接指向的资产目录。

`tools/materialize_workspace.py` 将 `source/` 复制到独立 `<ROOT>`，把原 `/mnt/ojc` 前缀对应到 `<ROOT>`，并显式映射环境与数据目录。默认只打印计划，加 `--apply` 才写入。不要选择原服务器项目目录作为目标。其他外部前缀通过 JSON 对象传入 `--path-map`；例如场景系统路径和共享内存工作目录需要按实际部署配置。

源码映射不是依赖安装，也不补齐候选清单。模型权重、第三方源码、已筛选输入、运行登记与历史结果均需分别提供。

## 2. 恢复依赖

见 [DEPENDENCIES.md](DEPENDENCIES.md)。刚体前端、FLUX、TRELLIS／ReconViaGen、渲染器及人体各阶段使用独立 Python 环境。历史实测环境约束保存在 `source/DriveLab/environments/`；它们是原主机版本记录，不是保证任意机器可直接解析的统一安装锁文件。CUDA 编译扩展须按本机 PyTorch／CUDA／编译器组合构建。

`source/DriveLab/requirements.txt` 是控制器 Python 包清单；不要据此把所有后端装到同一个环境。安装与第三方来源细节可对照 `source/DriveLab/docs/INSTALL.md`，其中相对链接对应其原 DriveLab 项目。

## 3. 刚体统一入口

在物化后的 `<ROOT>/DriveLab` 中编辑一份自己的 pipeline YAML。模板在 `configs/pipeline.example.yaml`、`configs/frontend.example.yaml`、`configs/generation.portable.json`。至少配置以下资源：

| 配置或环境变量 | 含义 |
|---|---|
| `NUSCENES_ROOT` / `source.path` | 原始数据集或帧／视频输入 |
| `GROUNDED_SAM2_ROOT` | Grounded-SAM-2 外部源码与权重 |
| `RIGID_PYTHON` | 检测／控制器 Python |
| `FLUX_PYTHON`, `TRELLIS_PYTHON`, `RENDER_PYTHON` | 各阶段独立解释器 |
| `FLUX_MODEL`, `H_MODEL`, `VGGT_MODEL`, `B0_MODEL` | 模型本地资源 |
| `TRELLIS_TORCH_HOME`, `BIREFNET_MODEL` | 附加模型缓存／分割模型 |
| `run_dir`, `library_dir` | 新批次目录与目标资产库目录 |
| `gpus`, `vlm_endpoint` | 本次可用 GPU 与实际 VLM 地址 |

```bash
cd /data/omnisim_workspace/DriveLab
python -m asset_extraction.run --help

# 创建源数据适配产物；这不是无副作用 dry-run。
python -m asset_extraction.run --config configs/my_pipeline.local.yaml --prepare-only

# 检测、分割和路由，不执行重建；需要模型与数据。
python -m asset_extraction.run --config configs/my_pipeline.local.yaml --handoff-only

# 在检查输入后执行该配置的完整构建。
python -m asset_extraction.run --config configs/my_pipeline.local.yaml
```

模板中的 `delete_rejected` 控制该新构建批次的拒绝输出是否回收。首次部署可设为 `false`，先保留输入与审查记录。不要直接对已有库运行 `compact_bank.py` 或历史 cleanup 入口。

## 4. 人体集成入口

在 `<ROOT>/DriveLab` 编辑 `human/config/runtime.example.json` 的本地副本。通过 `scripts/configure_human.py` 配置权重目录引用。先完成 vendor 来源恢复，不能仅创建空目录绕过检查。

```bash
cd /data/omnisim_workspace/DriveLab
export HUMAN_RUNTIME_CONFIG="$PWD/human/config/runtime.local.json"
export HUMAN_RUN_ROOT=/data/runs/omnisim_human_batch
export HUMAN_ASSET_ROOT=/data/asset_library/human
export NUSCENES_ROOT=/datasets/nuScenes
cd human
python run.py doctor
python run.py --help
```

`doctor` 会检查配置、解释器和模型资源，不启动生成；资源未配置时失败是预期行为，不能写成质量验收通过。nuScenes 使用 `intake` 子命令准备身份、真实曝光与动作；Waymo 入口消费已有筛选候选 CSV 和 G0 输入包。

新批次复制固定 `config/recipe.json`、`config/export.json` 与 `config/temporal_policy.json`（在运行目录名为 `TEMPORAL_POLICY.json`），再按 `human/README.md` 完成输入检查、部署记录和两例完整样本。后续执行顺序：

```bash
# 已有筛选包的 Waymo 批次；CSV 与 G0 文件需要实际准备。
python run.py batch init --limit 10 --pool /data/inputs/priority_candidates.csv
python run.py batch inspect --all --no-overviews

# 原生 nuScenes 输入。
python run.py intake rank --limit 10
python run.py intake prepare --cases HN0001 HN0002 --gpus 0 1
python run.py intake checks
python run.py intake review

# 在完成当前批次的实际输入及部署验证后。
python run.py batch pilot --cases HN0001 HN0002 --gpus 0 1
python run.py batch run --all --gpus 0 1
python run.py batch status
```

这组命令展示两种输入的对应入口，不应把 Waymo 初始化状态与 nuScenes 状态混写进同一个运行目录。源码整理不生成 `CODE_VALIDATION.json`、`BATCH_APPROVAL.json` 或视觉通过记录。

## 5. 与评测仓库连接

共同目录布局为：

```text
<ROOT>/
  DriveLab/
  workplace1/
    asset_library/                 # 真实库独立挂载/配置；不是源码发布内容
    human/
    4danyone/
    human_model_comparison/
    vehicle_model_comparison/
```

车辆仓库工作区准备工具使用 `--workspace <ROOT>/workplace1`；人体仓库使用 `HMC_WORKSPACE=<ROOT>/workplace1`。模型 Python 需要安装各自评测仓库的轻量路径适配包，按其 README 操作。真实已结算的实验资产与评价清单保持原引用，不通过这次源码整理重算或替换指标。

## 检查范围

本次发布验证源码完整性、语法、路径迁移和保护行为。未重跑 GPU 生成、Grid4D 优化、跨模型评测或完整数据集任务。历史 controller 的 GPU 编号、当前工作状态与机器路径仅反映原实验；新运行必须在本地配置后选择对应入口。
