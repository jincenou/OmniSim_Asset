from pathlib import Path
import json,hashlib,time
R=Path(__file__).resolve().parents[1]
cases=json.loads((R/'inputs/random10_cases.json').read_text());extra=json.loads((R/'docs/random10_additional_metrics.json').read_text())
labels=['蓝灰 SUV','白色皮卡','银色 MPV','白色 SUV','银色跨界 SUV','深蓝轿车','灰色 MPV','白色轿车','深色轿车','灰色电动 SUV']
main_table='''| 路线 | CLIP-I16 ↑ | DINO-I16 ↑ | ED-R ↓ |
|---|---:|---:|---:|
| B0 | 0.86774 | 0.54105 | 0.06088 |
| R | 0.89389 | 0.72170 | 0.02650 |
| F | 0.87985 | 0.54910 | 0.04413 |
| H | 0.90343 | **0.73880** | **0.02376** |
| B0＋1 real | 0.89353 | 0.66935 | 0.02895 |
| B0＋4 real | **0.90408** | 0.70535 | 0.02802 |
'''
head='''# Demo 入口

## 10 个新车辆 × 6 路线（已完成）

**60 组主实验＋10 组代表对象追加实验，共 70 个独立高斯 PLY 资产**。全部通过原 loader 重载、朝向复核、16 视角非空渲染和视频逐帧解码。十辆车各有一个六路线对照视频，另有三个追加实验视频。新批次尚未验证 SplatAD 场景插入。

B0＝FLUX 两图＋原 TRELLIS；R＝真实四图＋ReconViaGen；F＝FLUX 两图＋ReconViaGen；H＝真实四图和 FLUX 两图＋ReconViaGen；另外两条是在 B0 上加入 1／4 张真实图。B0 是显式双图对照，原生产车辆分支仅用侧图，不能把 B0 称为生产代码完全不变的复现。

主实验 seed=42，六路线 AOR 均为 **10/10（100%）**，三项源图指标共同有效子集为十个身份。

'''+main_table+'''
H 的 DINO-I16 和 ED-R 均值最好；B0＋4 real 的 CLIP-I16 略高。十对象不足以证明总体优胜，H 与 B0＋4 real 的三个配对差值置信区间均跨零。尺寸误差则是 F 最小，不能将外观指标等同于几何准确度。

[指标图](docs/random10_metrics.png) · [完整数值及置信区间](docs/random10_metrics.json) · [CSV](docs/random10_metrics.csv) · [宽高误差与成本](docs/random10_geometry_cost.json) · [报告](REPORT.md)

| 对象 | 六路线环绕视频 | 固定源视角 | H 高斯 PLY |
|---|---|---|---|
'''
for c,label in zip(cases,labels):
 p='results/'+c['case_id'];head+=f'| {label} | [视频]({p}/comparison_six_routes.mp4) | [对照]({p}/comparison_matched_source.jpg) | [PLY]({p}/H_seed42/asset/canonical.ply) |\n'
head+='''
[源视角总览 1—5](docs/random10_matched_overview_1.jpg) · [总览 6—10](docs/random10_matched_overview_2.jpg) · [全部 70 资产路径索引](docs/random10_asset_index.json)

## 代表白色皮卡：追加消融和精修

[1／2／4 张真实图与种子对照](results/random10_02_white_pickup/ablation_count_seed.mp4) · [原 TRELLIS 加真实图的种子对照](results/random10_02_white_pickup/trellis_real_assist_seed_comparison.mp4) · [R／F 官方精修前后](results/random10_02_white_pickup/refinement_R_F_comparison.mp4)

[追加指标 JSON](docs/random10_additional_metrics.json) · [CSV](docs/random10_additional_metrics.csv) · [追加审计](docs/random10_additional_audit.json)

R 精修的三项指标都改善；F 精修只改善 ED-R，CLIP-I16 和 DINO-I16 下降。真实图从 1 增至 2／4 张也非所有指标单调改善。以上结论仅对应一个代表身份。B0 相同 seed 重跑存在微小数值差异，文件哈希不相同，未宣称逐字节可复现。

## 输入、协议和审计

本批十个身份、每车四张真实输入和两张留出图均在生成前固定；20 张 FLUX 图全部重新生成。留出图未用于生成、精修、相机配准或尺度估计；当前相似度使用固定训练源图，并非留出视角泛化得分。FLUX 输入保留 `obj_id`、`image_path`、`category`、`prompts`，直接使用清单中的最终 prompt。

[新主矩阵](inputs/random10_experiment_matrix.json) · [追加矩阵](inputs/random10_additional_matrix.json) · [固定查询集](inputs/random10_query_registry.json) · [FLUX 交接清单](inputs/random10_flux_manifest.json) · [FLUX 总览](docs/random10_flux_overview.jpg) · [评估口径](docs/random10_evaluation_protocol.json) · [主审计](docs/random10_deliverable_audit.json) · [逐对象外观观察](docs/random10_quality_notes.json) · [代码及输入哈希](docs/random10_provenance.json)

本批是三个本地 Waymo 场景的小规模试验，原生 3D 元数据辅助关联、尺度与源相机；没有完整重跑 GroundingDINO／Qwen 属性分析，也不是论文 128 场景基准。DINO 使用经过校验的同架构 DINOv3 ViT-H+/16 timm 权重移植版，ED-R 的前景 patch 池化和匹配相机近似已明确记录。原 IDE 打开的 `inputs/experiment_matrix.json` 仍是最初批次，不代表本批进度。

'''
old=(R/'DEMOS.md').read_text();(R/'DEMOS.md').write_text(head+old[old.index('## 已交付的前几批实验'):])
section='''
## 新十对象：代表白色皮卡追加实验

十个追加资产均完成原 loader 重载、16 个非空轨道视角、匹配源视角、三项指标和视频解码。下表包含六个主实验参考行；不会把重复种子／精修当作新身份加入十对象均值。

| 路线 | seed | CLIP-I16 ↑ | DINO-I16 ↑ | ED-R ↓ |
|---|---:|---:|---:|---:|
'''
for x in extra['rows']:section+=f"| {x['variant']} | {x['seed']} | {x['CLIP_I16']:.5f} | {x['DINO_I16']:.5f} | {x['ED_R']:.5f} |\n"
section+='''
R 精修相对本次精修运行中保存的精确配对基线，CLIP-I16 +0.01277、DINO-I16 +0.01679、ED-R −0.00528；F 精修分别 −0.00251、−0.04476、−0.00281。精修没有统一改善所有指标；源视角图像优化也可能牺牲环绕一致性。R／F 精修本身分别用时 41.67／34.86 秒，峰值分配约 16.3 GiB；共享 GPU 负载下不作为公平速度比较。两者均执行官方相机注册及外观精修，无降级替代。配对基线在独立精修环境生成，与主实验 R/F 数值不同，因此只与各自 `*_refine_base` 配对。

R2 使用固定源图及训练集中方位差最大的另一图，R1 只用源图；R 使用四张训练图。双图相比单图提高环绕相似度，但 ED-R 略差；四图相比双图降低环绕相似度、改善 ED-R。视角覆盖及冗余比单纯数量更重要，但一个身份不能支持总体推断。

B0 同 seed 重跑的原始和规范化 PLY 哈希均不相同；CLIP-I16 差约 −0.00026、DINO-I16 −0.00047、ED-R −0.00001。保留实际输出，不能宣称位级确定性。R 的 seed 7 与 42 的 DINO-I16 相差 +0.06180，提示单种子排序存在波动。

[完整追加数值与哈希](docs/random10_additional_metrics.json) · [追加视频与文件入口](DEMOS.md) · [追加逐帧审计](docs/random10_additional_audit.json)。初次精修后处理曾错误寻找独立 `raw.ply`；已从本次保存的 `base_repeat.ply` 建立配对基线并完成导出，未重用主实验基线，也没有隐藏生成失败。

### 新批次的复用入口

工作目录为 `/mnt/ojc/workplace1/reconviagen_demo`。输入及查询已固定；不要重新抽样后覆盖当前基准。模型来源、脚本哈希及输入哈希见 `docs/random10_provenance.json`。已有指标可直接重汇总：

```bash
env/bin/python scripts/aggregate_asset_metrics.py --cases inputs/random10_cases.json --output random10_metrics
```

单个资产指标重新计算（不改生成结果；GPU 编号按实际资源设置）：

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/evaluate_asset_metrics.py --case random10_02_white_pickup --variants B0 R F H B0_real1 B0_real4 --encoder clip
CUDA_VISIBLE_DEVICES=0 /mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/evaluate_asset_metrics.py --case random10_02_white_pickup --variants B0 R F H B0_real1 B0_real4 --encoder dino
```

新十对象的原 loader 重载、非空视角和视频审计可用 `env/bin/python scripts/audit_random10.py` 与 `env/bin/python scripts/audit_random10_additional.py` 重跑。它们更新审计及规范化元数据，不重新生成资产。生成入口为 `scripts/run_variant.py`，精修入口为 `scripts/refine_case.py`（使用 `env_refine/bin/python`）；原批调度器只领取待运行任务，不自动覆盖完成项。原始生成、导出、朝向变更、评估和失败后修复日志全部保留在 `logs/`。
'''
p=R/'REPORT.md';txt=p.read_text();marker='\n## 新十对象：代表白色皮卡追加实验';p.write_text(txt.split(marker)[0]+section)
print('Updated DEMOS.md and REPORT.md')
