# 第三方来源与环境

本仓库不重复分发第三方 vendor、权重与人体模板。原业务代码需要的第三方修改不是任意最新版的上游代码；优先恢复下表的固定集成快照，再按该来源的声明使用。

| 依赖组 | 固定集成来源 | 应恢复的路径 |
|---|---|---|
| 刚体 TRELLIS B0、ReconViaGen 及本地适配 | `jincenou/DriveLab` commit `dde9b9d32c123cab17e1ffae94d5f6014418a500` | `<ROOT>/DriveLab/vendor/` |
| 集成人体 4DAnyone、GVHMR、Grid4D、RIFE、SAM2、FTGS 数据读取 | 同一 DriveLab commit | `<ROOT>/DriveLab/human/vendor/` |
| 原生产人体的固定 vendor | `jincenou/human_cvpr` commit `569d7f2886313062c0d4480d6e57388eda084037` | `<ROOT>/workplace1/human/vendor/` |
| 早期刚体前端 | `jincenou/DriveLab_pro_pipeline` commit `e13dcda2de30ca2e7bc27b4eb49f39c33c1b8bb8` | 已收录自有业务代码；历史外部运行环境另外配置 |
| 4DAnyone 上游 | `ant-research/4DAnyone` commit `9cc2aa230fe5d364da2c5634ec3dabadadcb0d60` | 历史路径 `<ROOT>/workplace1/4danyone/repo` |

上述 commit 是本地 checkout 的来源记录，不表示所有模型权重可从 Git 获得。DriveLab 自有业务文件的未提交修改已收录在 `source/`，以逐文件哈希为准，不能用 vendor 来源 commit 覆盖本仓库的业务代码。

可把指定版本的依赖仓库克隆到独立目录，只恢复对应 `vendor/` 子树到已物化的执行工作区。原目录中的权重链接需指向本机合法取得的权重；不要复制悬空链接后把部署检查视为通过。网络人体旧实验还有各自相机／骨架生成版本，详细参数以该阶段脚本和来源记录为准。

原上游来源说明：

- [刚体第三方声明](../source/DriveLab/THIRD_PARTY_NOTICES.md)
- [集成人体第三方声明](../source/DriveLab/human/THIRD_PARTY_NOTICES.md)
- [原人体第三方声明](../source/workplace1/human/THIRD_PARTY_NOTICES.md)
- [刚体固定 vendor 文件哈希](../provenance/rigid_vendor_hashes.json)
- [人体准备代码来源](../provenance/human_preparation_sources.json)
- [历史后端补丁与固定版本](../provenance/upstream_patches.json)

历史 Grid4D、4DGaussians、FreeTimeGsVanilla 与 Efficient4D 后端的本地变更以 `upstream_patches/` 中的补丁和少量新增适配文件保存。按 `provenance/upstream_patches.json` 检出相应基准 commit 后，先运行 `git apply --check <patch>`，再应用补丁；将 `overlay_files` 放到其中记录的依赖内相对路径。补丁仅表示本地改动，不包含外部仓库全文或编译产物。原第三方文件声明继续适用。

Grid4D 的 `base_commit` 是本地来源导入记录，不是远端可检出的提交；远端版本使用其单独列出的 `upstream_commit`（`3ab7b74482910d22fba7d51ac60c8cae1fe24bf1`）。本次补丁检查针对本地导入基准，恢复外部 checkout 后仍需执行 `git apply --check`。

主要模型包括 Grounded-SAM-2、FLUX、TRELLIS、ReconViaGen、4DAnyone、GVHMR、Grid4D、RIFE 与 BiRefNet。其权重、数据集、SMPL/SMPL-X、CUDA 光栅器和其他扩展按各自来源与许可配置。原版权声明继续适用，本整理不重新授予第三方代码统一许可。

`source/DriveLab/environments/*.constraints.txt` 保存分阶段环境版本；刚体控制器基础包在 `source/DriveLab/requirements.txt`。CPU 源码检查工具仅依赖 Python 标准库及 `bash`，不要求安装这些模型。
