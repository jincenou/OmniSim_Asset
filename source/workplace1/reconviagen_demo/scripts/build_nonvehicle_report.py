from pathlib import Path
import sys,json,csv
R=Path(__file__).resolve().parents[1];O=R/'nonvehicle10';sys.path.insert(0,str(R/'tools/report_deps'));import markdown
from docx import Document
from docx.shared import Pt,Inches
m=json.loads((O/'docs/metrics.json').read_text());cases=json.loads((O/'inputs/cases.json').read_text());qa=json.loads((O/'docs/visual_review.json').read_text())['reviews'];names=['绿灯','红灯','STOP 牌','限速30牌','圆筒垃圾桶','方形垃圾桶','白色消防栓','黄色消防栓','混凝土路障','橙色路锥'];variants=['B0','R','H','B0_real1']
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
fid={x['variant']:x['FID_2048'] for x in json.loads((O/'docs/fid_metrics.json').read_text())['results']}
main=table(['路线','AOR ↑','CLIP-I16 ↑','DINO-I16 ↑','PSNR-I16 ↑','SSIM-I16 ↑','LPIPS-I16 ↓','FID-2048 ↓'],[[s['variant'],f"{s['Nvalid']}/10",*[f"{s[k]['mean']:.4f}" for k in ['CLIP_I16','DINO_I16','PSNR_I16','SSIM_I16','LPIPS_I16']],f"{fid[s['variant']]:.2f}"] for s in m['summary']])
inventory=table(['实例','类别','真实图数','数据集'],[[n,c['category'],len(c['images']),c['dataset']] for n,c in zip(names,cases)])
routes=table(['路线','输入','生成后端'],[['B0','1张新 FLUX 图','原 TRELLIS，single-image'],['R','1–4张可靠真实观测','ReconViaGen main/v0.2'],['H','与R相同的真图 + 同一张FLUX','ReconViaGen main/v0.2'],['B0＋1 real','同一张FLUX + 用于生成它的原始真图','原 TRELLIS，multidiffusion，等权']])
selection=table(['对象','下一步选择／限制'],[[n,qa[c['case_id']]['selection']] for n,c in zip(names,cases)])
category=table(['类别','n','B0 DINO','R DINO','H DINO','B0+1 real DINO'],[[cat,next(x['n'] for x in m['category_summary'] if x['category']==cat),*[f"{next(x['DINO_I16'] for x in m['category_summary'] if x['category']==cat and x['variant']==v):.4f}" for v in variants]] for cat in sorted({c['category'] for c in cases})])
report=f'''# 非车辆资产：10个对象、4条路线、40组原生高斯实验

本批40组都完成了 Gaussian PLY 导出、原加载器重读、16视角非空渲染和指标计算。共有10个四路线环绕对比视频。**AOR=100%只表示能导出和渲染，不表示灯态、文字、几何或入库质量全部合格。**

总体上，R 的 CLIP、SSIM 与 LPIPS 外观诊断均值最好，H 的 DINO 均值最好；新增小样本FID也是R最低，仍没有统一胜者。就后续场景使用，普通路障、路锥及部分垃圾桶优先考察 H／B0+1 real；要求保持精确颜色的白色消防栓优先 R；本批两张路牌的四路线均存在文字或背面错误，不宜直接入库。

## 一、实际流程与样本

Waymo 原始 TFRecord → GroundingDINO SwinB 检测 → 人工核验实例 → 同相机 KLT 正反向跟踪及 RANSAC 短时关联 → SAM2.1 Hiera-Large 前景 → 白底参考 → FLUX.2 Klein → 四路线 Gaussian 生成 → 统一坐标导出与重新加载 → 环绕视频与评估。

路障和路锥来自 nuScenes 原始图像，使用原生实例 ID 关联、标注3D框投影裁剪与 SAM2；3D标注未用于重建输入。Waymo 不具备这些类别的统一对象位姿，本次没有编造姿态或米制尺寸。10例按类别多样性在生成前选定，是人工核验的 pilot，不是随机总体基准。

{inventory}

共33张可靠真实观测，STOP 的两张 SAM2 掩码失败、圆筒垃圾桶及其他若干帧关联失败，均在高斯生成前剔除；每条路线使用相同冻结清单中的输入。没有重复同一张图来凑多视图。多数 Waymo 观测只隔0.1–0.3秒，视角变化有限；因此本批 R 不能被理解为充分覆盖物体背面的真实多视图重建。没有独立 heldout 测试集。

FLUX 交接格式为 `obj_id / image_path / category / prompts`，prompts 数组直接传入原 wmj `flux._generate`。交通灯、路牌、路障、路锥沿用原类别提示词；垃圾桶、消防栓是新增类别，使用明确记录的补全及保色提示词。本批未重跑 Qwen 属性分析。每对象只生成1张 FLUX 图，不能与之前车辆的两张 FLUX B0 混淆。

## 二、四条路线的区别

{routes}

FLUX 为1024×1024、8步、guidance1.0。两种后端均使用 structure25步、latent25步、CFG5.0、interval[0.5,1.0]、rescale3.0、seed42。两者网络和条件融合机制不同；R/H与B0之间的差异包含后端变化。B0+1 real 相对B0则是在同一原TRELLIS后端增加真实证据。

所有结果保留原生 Gaussian，未进行 mesh→Gaussian 转换。原始文件保留为 raw.ply；如 opacity logit 存在正负无穷，仅执行已有饱和值等价修复（激活后变化≤1e-7），再按原序列化器导出 canonical.ply。z-up、水平居中、底部对齐；最长稳健范围归一化为2个任意单位，**不是2米**。没有从源图搜索最佳朝向，保留后端 yaw；语义朝上仍可能失败，例如绿灯的B0+1 real横放。

这是通用 TRELLIS 对照，不是原工程针对路牌的专用平面重建分支；不能据此否定那个未测试的专用分支。

## 三、指标结果

{main}

共同有效身份为10例；所有方法同一集合。不同类别只占1或2例，类别比较仅用于定位问题：

{category}

FID为2026-10-23新增复测，复用原冻结10例和既有640张渲染，未重新生成资产；原CLIP/DINO和像素指标保持原计算值并复核聚合。**这里FID是整个集合的单个分布距离，不是逐对象分数的平均。**

### 指标怎样计算

- AOR：满足非空、可导出、有效Gaussian属性、原加载器重读成功、16个环绕视图均有前景的实例数 / 10。薄片、横放、错字、错误灯态不会自动成为 AOR 失败。
- CLIP-I16／DINO-I16：每个对象固定 real_00 的源前景，分别与16张每隔22.5°的渲染前景计算归一化特征余弦相似度；先平均16视图，再平均10个身份。CLIP为OpenAI ViT-B/32；DINO为DINOv3 ViT-H+/16 qkvb，CLS特征，float32，严格加载同一已审计权重。每图前景独立紧框、每边5%方形留白、白底512；CLIP再走标准224预处理。
- PSNR-I16：每视图 RGB sRGB[0,1] 全图MSE，算 -10log10(MSE)，先平均对象的16个dB值，再平均身份。SSIM-I16：skimage，高斯窗口11、sigma1.5、总体协方差、RGB均值。LPIPS-I16：官方AlexNet、v0.1、RGB[-1,1]；同样先视图后身份平均。
- 这些像素指标比较的是同一源图与全部环绕视图，独立前景归一化，没有像素几何对应。**它们是外观诊断，不是准确重建误差；不是之前车辆报告的表A／表B，不能合并横比。**背面没有真实图，白底也影响结果；补充CSV保留前景联合区域诊断。
- FID-2048：采用 [pytorch-fid 0.3.0](https://github.com/mseitzer/pytorch-fid) 的官方FID兼容Inception v3 pool3（2048维、float32、关闭TF32）。固定10张real_00源前景为参考，每条路线10×16=160张环绕前景为生成集合。沿用白底512前景，再由该实现双线性缩放到299、映射到[-1,1]；不对2048维特征作L2归一化。分别计算特征均值和ddof=1协方差，FID=均值差的平方范数+两协方差迹−2倍协方差乘积平方根的迹。实际用float64低秩SVD等价计算，并与上游SciPy实现在实际2048维数据上交叉校验。
- **FID越低越好，但这里只有10个参考身份，协方差秩至多9，存在强小样本偏差；160张环绕图不是160个独立物体，源图与背面渲染也不构成同视角分布。**仅适用于本批相同样本/视角/预处理下的探索性比较，不能拿数值与其他论文直接比，也不能作为几何、灯态、文字正确或场景插入自然的证明。没有计算单对象或类别FID。完整输入哈希、权重来源和数值核验见fid_protocol.json、fid_metrics.json。
- ED-R 本批记为 N/A。没有独立标定的源图到生成物体的相机变换，不能用事后最好看的视角冒充 matched-source 评估。即使 nuScenes 两例有原始框姿态，也没有对应生成资产的可靠正面约定。

10例的配对bootstrap（10000次，seed20261023）中，H-R 的 DINO 差为+0.0293，95%区间约[-0.0494,+0.1335]；CLIP差为-0.0117，区间约[-0.0303,+0.0028]。小样本下不能据均值宣称H或R总体显著胜出；样本也是人为选取的多样性pilot。

## 四、可视化发现与选择

{selection}

最有说明力的失败链：绿灯原图→FLUX多亮一盏灯→B0继承错误；白色消防栓原图→FLUX变红→H正面灰白但背面红；限速牌原图SPEED→FLUX拼成SOERD→所有通用生成路线仍有字形问题。R可保留更多源外观，却可能生成过薄的交通灯或路障；更高的源图相似度并不保证三维实体合理。

建议汇报顺序：07白色消防栓（颜色与多视角冲突）→01绿灯（语义与形体分离）→10路锥（H的补形效果）→09路障（薄片问题）→03/04路牌（不能只看特征分数）。10例全部保留，不按结果挑选统计样本。

这批非车辆只完成独立资产实验，**尚未插入重建场景**。同期完成的场景验证使用的是之前生成的SUV与皮卡；不要将其当作非车辆插入效果的证明。真实尺寸、支架、牌面内容、灯态和入库筛查仍应逐对象完成。

## 五、文件入口

- `results/<对象>/comparison_video.mp4`：四路线6秒环绕视频，左侧原始裁图与FLUX参考。
- `results/<对象>/comparison_orbit.jpg`：原图、FLUX及四路线0/90/180/270度对比。
- `results/<对象>/<路线>_seed42/asset/canonical.ply`：可独立加载的高斯文件。
- `inputs/<对象>/source_full.jpg`：原始数据集全帧预览；`real_00.png`：保留的源裁图。
- `docs/summary.csv`、`per_asset_metrics.csv`、`per_category_metrics.csv`、`metrics.json`、`paired_differences.json`：可核对表格与区间。
- `docs/evaluation_protocol.json`、`input_visual_audit.json`、`visual_review.json`、逐对象manifest和生成日志：评估口径、失败记录、来源与可复现参数。
'''
(O/'REPORT.md').write_text(report);(O/'REPORT.txt').write_text(report);css='body{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1200px;margin:auto;padding:28px;line-height:1.65}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #ccc;padding:8px}th{background:#edf3f9}img,video{max-width:100%}code{background:#eee}section{margin:35px 0}'
body=markdown.markdown(report,extensions=['tables','sane_lists']);(O/'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><title>非车辆四路线实验报告</title><style>'+css+'</style>'+body)
doc=Document();doc.add_heading('非车辆40组实验汇报',0)
lines=report.splitlines();i=0
while i<len(lines):
 line=lines[i]
 if line.startswith('|'):
  block=[]
  while i<len(lines) and lines[i].startswith('|'):block.append([x.strip() for x in lines[i].strip('|').split('|')]);i+=1
  t=doc.add_table(rows=1,cols=len(block[0]));t.style='Table Grid'
  for j,x in enumerate(block[0]):t.rows[0].cells[j].text=x
  for row in block[2:]:
   cells=t.add_row().cells
   for j,x in enumerate(row):cells[j].text=x
  continue
 if line.startswith('#'):doc.add_heading(line.lstrip('# ').strip(),min(len(line)-len(line.lstrip('#')),3))
 elif line:doc.add_paragraph(line.replace('**','').replace('`',''))
 i+=1
doc.save(O/'REPORT.docx')
html='<!doctype html><meta charset="utf-8"><title>十个非车辆demo</title><style>'+css+'</style><h1>10个非车辆 × 4路线</h1><p><a href="REPORT.html">完整汇报</a> · <a href="REPORT.docx">Word</a> · <a href="docs/summary.csv">指标CSV</a></p>'+markdown.markdown(main,extensions=['tables'])+'<p>FID为10源图对160环绕图的小样本分布诊断；像素指标为环绕外观诊断；AOR不代表语义正确；ED-R未计算。详细限制见报告。</p>'
for name,c in zip(names,cases):
 cid=c['case_id'];html+=f'<section><h2>{name} · {cid}</h2><p>{qa[cid]["selection"]}</p><video controls preload="none" src="results/{cid}/comparison_video.mp4"></video><p><a href="inputs/{cid}/source_full.jpg">全帧原图预览</a> · <a href="inputs/{cid}/real_00.png">源裁图</a> · <a href="inputs/{cid}/flux_00.png">FLUX图</a></p><img loading="lazy" src="results/{cid}/comparison_orbit.jpg"></section>'
(O/'START_HERE.html').write_text(html);print('REPORT COMPLETE',flush=True)
