"""Report frozen feasible comparisons; never invent missing ground truth scores."""
from pathlib import Path
import csv
import hashlib
import json
import statistics
R=Path(__file__).resolve().parents[1];B=R/'benchmark'
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 manifest=read(B/'manifest.json');protocol_sha=sha(B/'protocol.json');rows=[];results={}
 for job in manifest['jobs']:
  p=B/'results'/job['id']/'metrics.json'
  if not p.exists():continue
  d=read(p)
  if d['protocol_sha256']!=protocol_sha or d['evaluated_images']!=363:raise ValueError('Protocol mismatch')
  if d['job']!=job:raise ValueError('Result does not match frozen manifest')
  results[job['id']]=d;rows.append({'label':job['label'],'recipe':job['recipe'],'steps':d['steps'],'gaussians':d['gaussians'],**d['aggregates'],'path':str(p)})
 pairs=[]
 for track in manifest['tracks']:
  name=track['label'];q0=results.get(name+'_q0');q50=results.get(name+'_budget_half_50k');dense=results.get(name+'_capacity50k')
  if q50 and dense:
   if not q0 or q0['inputs']!=q50['inputs'] or q50['inputs']!=dense['inputs']:raise ValueError('Paired reference pixels differ')
   pair={'label':name,'role':track['role'],'q0':q0['aggregates'],'q50':q50['aggregates'],'dense50':dense['aggregates'],
    'q0_to_q50':{k:q50['aggregates'][k]-q0['aggregates'][k] for k in ['full_psnr','ssim','lpips']},
    'q50_to_dense50':{k:dense['aggregates'][k]-q50['aggregates'][k] for k in ['full_psnr','ssim','lpips']},
    'lpips_relative_reduction_pct':100*(q50['aggregates']['lpips']-dense['aggregates']['lpips'])/q50['aggregates']['lpips']}
   pairs.append(pair)
 summary={'state':'complete' if len(rows)==len(manifest['jobs']) else 'partial','distinct_baselines_scored':sum(x['recipe']=='q0' for x in rows),'completed_jobs':len(rows),'requested_jobs':len(manifest['jobs']),'protocol_sha256':protocol_sha,'rows':rows,'paired_results':pairs,'real_ground_truth_metrics':'not_computed_missing_registered_real_novel_views','visual_approval':False,'independent_validation_deltas':{k:statistics.mean(p['q50_to_dense50'][k] for p in pairs if p['role']=='validation') for k in ['full_psnr','ssim','lpips']} if any(p['role']=='validation' for p in pairs) else {}}
 write(B/'results.json',summary)
 if rows:
  with (B/'results.csv').open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
 lines=['# Waymo 4DAnyone → FreeTimeGsVanilla 基线对照','',
 '本轮按最新要求，只做现有数据能够支持的比较。缺少真实新视角真值的两类论文指标不计算，不另外启动仅为凑齐论文设置的实验。',
 '',f'当前已完成 **{summary["distinct_baselines_scored"]}/8 条基线、{len(rows)}/16 项评测**。每项使用 3 个留出生成视角 × 全部 121 帧，共 363 对图像。8 条不同身份中，4 条有原 5k、原容量 50k、增加容量 50k 的配对；另 4 条用于动作/失败覆盖，尚无全部优化分支。',
 '', '## 链路与基线边界','',
 'Waymo 真实人物轨迹 → 裁剪/插值/实例分割 → GVHMR → 官方 4DAnyone-Turbo 公开权重与生成模块 → 24 路同步视频 → 每视角 RGBA/相机/时间 → 训练视角轮廓初始化 → 独立 FreeTimeGsVanilla 优化 → 动态渲染、checkpoint 和 .ftgs.ply。',
 '', '这里的 baseline 是当前服务器已运行的 Waymo 配方：预处理、静态相机假设和后端桥接含本地适配。行走 walk063 使用 root-local 改动，单独标注，不声称未经修改的官方端到端链路。前端没有重新训练；后端是逐资产优化。',
 '', '论文三类评测均采用 PSNR、SSIM、LPIPS：生成视频对真实视频、重建渲染对真实视频、留出生成视角一致性。当前只实施第三类的本地版本。论文主要评测使用 DNA-Rendering/DyMVHumans、16 视角/98 帧，另有 121 帧消融；本文使用 Waymo、24 视角/121 帧，不能与论文表格数值直接排名。[论文 §4.1 与附录 F](https://arxiv.org/html/2608.20335#S4)。',
 '', '作者的 FreeTimeGS 实现未在官方重建教程发布；现用独立 FreeTimeGsVanilla 不能冒称作者后端。[官方重建说明](https://github.com/ant-research/4DAnyone/blob/main/docs/nerfstudio.md)。',
 '', '## 固定评测口径','',
 '- 留出视角 0/8/16，训练使用另外 21 个视角；每个留出视角全 121 帧。',
 '- 352×640 原有 RGBA 监督母版，统一白背景；整图 PSNR、SSIM、AlexNet LPIPS，无裁剪、无缩放、无后处理滤点。SSIM 采用 Gaussian 11、σ=1.5、data_range=1。',
 '- 同时记录前景 PSNR、IoU、mask 外 alpha；白背景可能稀释人体误差，所以这些诊断和视频必须一起看。预测分割不是人工真值。',
 '- 配对结果检查全部参考图 SHA256、检查点 SHA256 和协议 SHA256。每图指标取均值，身份间等权；不会把相关视频帧当成独立受试者做显著性声明。',
 '- 增加容量比较均为 50k；5k→50k 同时改变学习率计划，是新训练预算实验，不是假装从 5k 严格续训。',
 '- 单次固定种子仅报告描述性收益；接近噪声量级的小提升需再做相同身份/多个种子的配对复核。没有统一的“提高多少就肉眼可见”阈值。',
 '', '## 8 条现有基线','',
 '|身份|动作/困难情况|PSNR ↑|SSIM ↑|LPIPS ↓|效果|','|---|---|---:|---:|---:|---|']
 for t in manifest['tracks']:
  d=results.get(t['label']+'_q0');v=d['aggregates'] if d else None
  scores=f'{v["full_psnr"]:.3f}|{v["ssim"]:.4f}|{v["lpips"]:.4f}' if v else '待完成|待完成|待完成'
  lines.append(f'|{t["label"]}|{t["behavior"]}|{scores}|[四联屏]({t["demo"]})|')
 lines+=['','动作标签来自已有抽样复核；0084 仍标注“迈步/转身候选”，不把它计为已完成全片动作审核的第二条持续行走。低质量和失败身份保留，不以指标好看为由剔除。',
 '', '## 50k 原容量 → 50k 增加容量','',
 '|身份|角色|ΔPSNR dB ↑|ΔSSIM ↑|ΔLPIPS ↓|LPIPS 相对下降|','|---|---|---:|---:|---:|---:|']
 for p in pairs:
  v=p['q50_to_dense50'];lines.append(f'|{p["label"]}|{p["role"]}|{v["full_psnr"]:+.3f}|{v["ssim"]:+.4f}|{v["lpips"]:+.4f}|{p["lpips_relative_reduction_pct"]:.2f}%|')
 lines+=['','完整 5k/50k/增加容量数值及所有逐帧记录见 [results.json](results.json)、[results.csv](results.csv)。旧版灰背景前景 PSNR/裁剪 LPIPS 与本表口径不同，不能混成一列。',
 '', '## 如何据此决定下一步','',
 '生成视频已出现头发、衣服、附件或动作错误时，重建指标提升可能只是更忠实地拟合错误，优先查生成条件和输入。生成正常但高斯模糊时，检查容量、监督与时间表示；原容量 50k 与增加容量 50k 的配对用于区分单纯长训和容量变化。用同相机固定时间轴对照，并另外看环绕几何、手脚/附件与漂浮点。',
 '', '先复核两条独立身份的收益与失败，再把确定有价值的配方扩展到行走身份；移动相机适配是另一条受控实验，不与容量调整一起改动后归因。暂不据此恢复 100 条批量资产生产。',
 '', '## 后台与可追溯记录','',
 '使用独立 tmux 服务 fdwaymo 的 round2_benchmark，会话随作业结束退出。仅物理 GPU 4–7，允许与其他项目共享，按可用显存准入。',
 '', '[固定样本清单](manifest.json) · [评测协议](protocol.json) · [实际队列状态](queue.json) · [全部 demo 页](../gallery/index.html)。历史准备和配置检查失败保留在 failed_split_config/，成功结果不覆盖失败记录。']
 (B/'REPORT.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'state':summary['state'],'jobs':len(rows),'pairs':len(pairs)}))

if __name__=='__main__':main()
