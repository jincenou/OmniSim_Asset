from pathlib import Path
import sys,json,csv,shutil,re,html,hashlib
R=Path(__file__).resolve().parents[1];O=R/'deliveries/vehicle_experiments_20261023';sys.path.insert(0,str(R/'tools/report_deps'))
import markdown
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
D=json.loads((R/'docs/random10_metrics.json').read_text());P=json.loads((R/'docs/random10_pixel_metrics.json').read_text());G=json.loads((R/'docs/random10_geometry_cost.json').read_text());E=json.loads((R/'docs/random10_additional_metrics.json').read_text())
V=['B0','R','F','H','B0_real1','B0_real4'];labels={'B0_real1':'B0＋1 real','B0_real4':'B0＋4 real'}
def label(v):return labels.get(v,v)
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(str(x) for x in row)+' |\n' for row in rows)
def pix(mode,v):return next(x for x in P['summary'] if x['alignment']==mode and x['variant']==v)
def emb(v):return next(x for x in D['summary'] if x['variant']==v)
def geo(v):return next(x for x in G['summary'] if x['variant']==v)
main=table(['路线','AOR ↑','CLIP-I16 ↑','DINO-I16 ↑','ED-R ↓'],[[label(v),'100%',*[f'{emb(v)[k]["mean"]:.5f}' for k in ['CLIP_I16','DINO_I16','ED_R']]] for v in V])
pix_tables={m:table(['路线','PSNR/dB ↑','SSIM ↑','LPIPS-Alex ↓','轮廓 IoU ↑'],[[label(v),*[f'{pix(m,v)[k]["mean"]:.4f}' for k in ['PSNR_dB','SSIM','LPIPS_alex','silhouette_IoU']]] for v in V]) for m in ['normalized','native']}
geom=table(['路线','宽度 MAPE/% ↓','高度 MAPE/% ↓','推理均时/s','峰值分配/GiB'],[[label(v),*[f'{geo(v)[k]:.2f}' for k in ['width_MAPE_percent','height_MAPE_percent','inference_seconds_mean','peak_allocated_GiB_mean']]] for v in V])
refrows=[]
for v in ['R_refine_base','R_refined','F_refine_base','F_refined']:
 a=next(x for x in E['rows'] if x['variant']==v);b=next(x for x in P['rows'] if x['variant']==v and x['alignment']=='normalized');refrows.append([v,f'{a["DINO_I16"]:.4f}',f'{a["ED_R"]:.4f}',f'{b["PSNR_dB"]:.2f}',f'{b["SSIM"]:.4f}',f'{b["LPIPS_alex"]:.4f}'])
refine=table(['配对实验','DINO-I16 ↑','ED-R ↓','PSNR ↑','SSIM ↑','LPIPS ↓'],refrows)
union=table(['路线','并集前景 PSNR/dB ↑','并集前景 SSIM ↑'],[[label(v),f'{pix("normalized",v)["PSNR_union_dB"]["mean"]:.4f}',f'{pix("normalized",v)["SSIM_union"]["mean"]:.4f}'] for v in V])
report='''# 车辆高斯资产生成实验汇报

2026 年 10 月 23 日｜10 个新车辆 × 6 条主路线｜代表对象追加消融与精修

## 一、汇报结论与建议

**建议将 H（4 张真实图＋2 张 FLUX 图 → ReconViaGen）作为下一阶段研究主线，保留 R 作为必须对照的强基线；如果工程上要求继续使用原 TRELLIS，则采用 B0＋4 real。**这是一项基于目标取舍的建议，不是“所有指标 H 都最好”。

推荐 H 的理由是：本任务希望同时保留原实例身份、支持多视角观看并兼顾整体投影。H 的 DINO-I16 和 ED-R 均值最好；在保留相机投影位置的补充评估中，PSNR、LPIPS、轮廓 IoU 的均值也最好。它相对纯真实图 R 的宽度误差有所下降，但仍明显偏大。FLUX 必须通过身份一致性检查，不能无条件混入。

新增评估也显示 H 的局限：前景居中归一化后，R 的 PSNR、SSIM、LPIPS 和轮廓 IoU 均值全部更好。因此，若目标只强调源视角局部外观还原，R 更值得选择。H 与 B0＋4 real 的多项差异区间跨零，十个对象不足以认定普遍优胜。本次不把不同量纲指标强行加权为一个总分。

## 二、实验问题与规模

实验回答三个问题：真实多图能否降低 FLUX 身份漂移；原 TRELLIS 加真实图能否得到同类收益；多图数目、随机种子和官方精修是否稳定改善结果。

从三个本地 Waymo 场景的随机候选流中按输入质量筛选并固定 10 个新车辆，抽样 seed=20261023。每车预先固定 4 张真实训练图和 2 张留出图，重新生成 2 张 FLUX 图。所有主路线使用 seed=42。共 60 个主资产，代表白色皮卡追加 10 个资产，共 70 个高斯 PLY。所有资产完成原 loader 重载、朝向复核、16 个非空视角与视频解码；十对象新批次尚未做 SplatAD 场景插入。

留出图没有参与生成、精修、尺度或相机配准；本报告的源条件指标使用固定训练源图，不是留出视角重建指标。代表对象重复种子和精修只作消融，不作为新身份加入十对象均值。

## 三、实际 pipeline 与六条路线

![实际实验流程](images/overview/pipeline.png)

实际流程为：原生 Waymo 标签／3D track 辅助选对象和关联 → 人工复核真实视图 → SAM2 分割 → 固定源图白底前景 → 原 FLUX.2 Klein 实现生成可见端面及侧面 → 分路线进行三维生成 → 导出高斯 PLY → 统一 z-up、车头 +X、接地和长度尺度 → 原 loader 重载 → 环绕及固定源相机渲染 → 指标与人工审核。

本批没有重新执行 GroundingDINO 和 Qwen 属性分析；直接沿用原 FLUX 可见端面／侧面提示词。交接清单保留 obj_id、image_path、category、prompts，prompts 直接传入原 flux.py::_generate。FLUX 设置为 1024×1024、8 步、guidance=1.0；输出采用 BiRefNet-lite 去背景。不能把这一 pilot 说成论文全流程已原样复现。

'''+table(['路线','三维模型输入','后端','隔离的问题'],[
['B0','同一对 FLUX 端面＋侧面','原 TRELLIS，多图 multidiffusion','双图基准'],
['R','4 张真实分割图','ReconViaGen / TRELLIS-VGGT v0.2','直接使用真实多视图'],
['F','与 B0 相同的 2 张 FLUX 图','ReconViaGen / TRELLIS-VGGT v0.2','固定输入，替换后端'],
['H','与 R 相同的 4 张真实图＋同一对 FLUX 图','ReconViaGen / TRELLIS-VGGT v0.2','真实证据和生成参考混合'],
['B0＋1 real','FLUX 两图＋固定真实源图','原 TRELLIS，多图等权','少量真实图辅助'],
['B0＋4 real','FLUX 两图＋与 R 相同的四图','原 TRELLIS，多图等权','保留原后端，增加真实证据']])+'''
原 TRELLIS 路线通过 multidiffusion 融合各图条件，未为真实图设置额外权重。ReconViaGen 路线使用固定版本的 TRELLIS-VGGT 多视图条件生成。两类后端共享 25 structure＋25 latent 步、CFG 5.0、区间 [0.5,1.0]、rescale 3.0；使用同一批源图和评估口径，但输入图数与后端前处理不同，不是等算力比较。

注意：旧生产车辆代码仅使用 FLUX 侧图；这里 B0 是原后端的显式双图对照，不等于旧生产配置完全不变。

## 四、怎么评估

AOR 是有效输出数／固定 10 个查询：高斯文件非空、属性合法、可独立加载，且 16 个环绕视角都有非空前景。它衡量可用输出率，不代表外观全部合格。

CLIP-I16／DINO-I16：对每辆车，固定一张真实源图，分别与 16 个环绕渲染图计算特征余弦相似度，再先平均视角、后平均对象。相邻角度 22.5°，统一仰角和渲染设置。CLIP 为 OpenAI ViT-B/32；DINO 为 DINOv3 ViT-H+/16 timm 移植权重的 CLS 特征。I16 的 16 指视角数量。

ED-R：根据原生相机内外参和车辆姿态渲染固定源视角，不从环绕图里挑最高分。源图与渲染图独立居中、等比例归一化；用 DINOv3 前景 patch 平均特征计算 1−余弦相似度，越低越好。它与 DINO-I16 的视角和池化方式不同。

新增 PSNR／SSIM／LPIPS 使用同一固定源视角，预先固定两套口径，完整保留结果：

- 主表 A：沿用 ED-R 的前景居中归一化；正方形两侧各留 5% 边距、白底、512×512。消除平移和整体大小差异，主要看归一化后的外观与轮廓。
- 补充表 B：保留原始裁剪相机画布中的投影位置、大小；两图施加同一个补方和缩放。没有各自居中，不优化姿态，因而对尺度和相机近似更敏感。
- PSNR：RGB [0,1] 的像素均方误差，PSNR=10log10(1/MSE)，单位 dB；先逐对象算 dB，再取均值。没有颜色／曝光拟合。
- SSIM：RGB 三通道平均，data_range=1、Gaussian sigma=1.5、11×11 窗、population covariance。
- LPIPS：官方 AlexNet、学习校准 v0.1，RGB 归一化至 [−1,1]，float32、eval 模式；标准整张方形图的空间平均。来源：[LPIPS 官方实现](https://github.com/richzhang/PerceptualSimilarity)。
- 白底可能抬高整图 PSNR／SSIM，因此同时报告两图前景掩码并集上的敏感性结果。并集指标是明确标注的补充实现，不冒充标准 SSIM；不只取掩码交集来回避缺失部分。

所有主表使用六路线共同有效的同一批十身份。置信区间按身份 bootstrap 10,000 次，不把视角当成独立样本。源相机使用单时刻近似、未建模逐行滚动快门；PSNR 等尤其不能解释为严格新视角重建精度。

![两套对齐口径示例：04 白色 SUV](images/overview/alignment_example.jpg)

## 五、主实验结果

### 5.1 源条件特征指标

'''+main+'''
H 的 DINO-I16 与 ED-R 均值最优；B0＋4 real 的 CLIP-I16 略高。真实证据的加入比只使用 FLUX 更能保持实例外观。H 相对 B0＋4 real 的三个特征指标差值的配对置信区间均跨零，不能据此宣称稳定优胜。

### 5.2 表 A：前景归一化后的固定源视角

'''+pix_tables['normalized']+'''
R 的四项均值最佳。H−R 的 LPIPS 差值为 +0.01204，描述性配对 bootstrap 95% 区间 [0.00665,0.01724]；SSIM 差值 −0.00877，区间 [−0.01543,−0.00172]。这说明混入 FLUX 可能牺牲部分源图局部外观保真，不能仅据旧的特征指标认定 H 全面最好。

### 5.3 表 B：保留相机投影位置与大小

'''+pix_tables['native']+'''
此口径 H 的 PSNR、LPIPS、轮廓 IoU 均值最佳，B0＋4 real 的 SSIM 最佳。与表 A 排名不同，说明几何比例和投影偏差会改变评价；两表不能混成一个得分，也不能选择性只汇报有利的一表。

![两套像素指标及置信区间](images/overview/pixel_metrics.png)

### 5.4 几何比例与实测成本

'''+geom+'''
宽高误差根据规范化后高斯中心的 0.5%—99.5% 分位范围计算；各路线仅按同一输入框长度等比例缩放，再与训练观测原生框的中位宽高比较。不是完整表面 GT，也不是逐轴拉伸后的结果。F 的宽高误差最小，但外观身份保持较弱；R 的宽度误差最大。耗时仅包含模型推理，排除加载、SAM2、FLUX、导出和渲染，并受共享 GPU 负载影响，不作受控性能排名。

## 六、代表对象消融与精修

白色皮卡追加 R1／R2、R 的第二 seed、两条原 TRELLIS 真实辅助路线的第二 seed、B0 同 seed 重跑，以及 R／F 精修及各自同运行基线，共十个资产。R2 为固定源图＋方位差最大的另一张训练图；没有用重复图凑八图。

'''+refine+'''
精修前均来自本次精修运行保存的精确配对基线，不与另一环境的较早主实验输出混配。官方流程包括输入配准和外观梯度优化，ss_refine=No。R 精修改善上述源视角指标及 DINO／ED-R；F 精修只轻微改善 LPIPS／ED-R，DINO、SSIM、PSNR 没有同步改善。这个单对象结果不支持全量默认启用精修；由于属于源条件优化，源图得分提升也不能当作未见视角改善证据。

R 从一图到双图提高了环绕特征相似度，但源视角像素指标下降；从双图到四图也不呈现所有指标单调改善。R 的 seed 7 与 42 的 DINO-I16 相差约 0.0618。B0 相同 seed 重跑的 PLY 哈希不同、指标差异很小，不能宣称逐字节确定性。

## 七、可视化解读与展示顺序

建议先展示 04 白色 SUV 的原图和七列对比，说明真实图能减少 FLUX 前脸、车轮与车身比例漂移；再看 02 白色皮卡的六路线环绕视频，展示三维完整性和纹理差异。然后展示 10 灰色电动 SUV：FLUX 生成了燃油车式前格栅，B0/F 继承了身份漂移，真实条件路线更接近源车。这是机制说明，不是挑选这三例计算指标；十例完整结果都在包内。

最后主动展示 08 白色轿车：真实输入主要只有后侧，R 未观测车头出现前后相似的尾部纹理，车头标注置信度为 0.5；高源视角分数不能验证未见面。07 灰色 MPV 的 H 也有尾部前脸重复纹理。所有资产保持待审核，不宣称生产质量通过。

图片列顺序统一为 Source、B0、R、F、H、B0_real1、B0_real4。所有 comparison_matched_source.jpg 均从实验目录原样复制，未为汇报重新挑角度。完整原始相机 JPEG、未分割源裁剪、分割前景和目标位置标记图均提供；JPEG 从对应 TFRecord 提取，裁剪像素与已冻结源图核对一致。

## 八、下一步决策

**当前优先推进 H，保持 R 与 B0＋4 real 的同对象对照。**先修真实视图筛选与配准、过滤 FLUX 身份漂移，再评估是否值得保留混合输入。若必须马上在原 TRELLIS 工程上落地，选择 B0＋4 real，不以替换后端为前提；只看源图局部外观的任务选择 R。

下一轮应扩展身份与随机种子，并单独建立可靠留出视角评估；在统一真实尺度下核查宽高和场景接地。精修仅作为可选实验，不默认开启。最终推荐是针对本轮任务的研究选择，不是以十对象证明普遍算法优胜。

## 附录：前景并集敏感性检查

下面使用表 A 的归一化图，仅在两图前景掩码的并集内汇总，减少共同白底贡献。R 的并集 PSNR 最好；B0＋4 real 的并集 SSIM 最好，再次说明没有一条路线在所有指标上占优。

'''+union+'''
原始逐对象数据、两套对齐结果、置信区间和实现口径位于 metrics/；代码及来源位于 reproducibility/。压缩包用于展示，不包含大体积高斯 PLY、模型权重和环境。全部 70 个 PLY 的服务器路径保存在 reproducibility/asset_paths.json，仍位于原实验目录。
'''
# Human-editable document with actual tables and embedded figures.
doc=Document();sec=doc.sections[0];sec.top_margin=Inches(.65);sec.bottom_margin=Inches(.65);sec.left_margin=Inches(.65);sec.right_margin=Inches(.65)
normal=doc.styles['Normal'];normal.font.name='Calibri';normal.font.size=Pt(10);normal._element.rPr.rFonts.set(qn('w:eastAsia'),'Microsoft YaHei');normal.paragraph_format.space_after=Pt(6)
for style in ['Title','Heading 1','Heading 2','Heading 3']:
 doc.styles[style]._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
def plain(s):return re.sub(r'\[([^\]]+)\]\([^)]+\)',r'\1',s).replace('**','').replace('`','')
lines=report.splitlines();i=0
while i<len(lines):
 line=lines[i]
 if not line.strip():i+=1;continue
 if line.startswith('|'):
  trs=[]
  while i<len(lines) and lines[i].startswith('|'):
   cells=[plain(x.strip()) for x in lines[i].strip('|').split('|')]
   if not all(re.fullmatch(r'[-: ]+',x) for x in cells):trs.append(cells)
   i+=1
  t=doc.add_table(rows=1,cols=len(trs[0]));t.style='Light Shading Accent 1'
  for j,x in enumerate(trs[0]):t.rows[0].cells[j].text=x
  # Repeat table header on later pages.
  trPr=t.rows[0]._tr.get_or_add_trPr();repeat=OxmlElement('w:tblHeader');trPr.append(repeat)
  for row in trs[1:]:
   cells=t.add_row().cells
   for j,x in enumerate(row):cells[j].text=x
  for row in t.rows:
   for cell in row.cells:
    for p in cell.paragraphs:
     for run in p.runs:run.font.size=Pt(8)
  continue
 if line.startswith('!['):
  path=re.search(r'\]\(([^)]+)\)',line).group(1);doc.add_picture(str(O/path),width=Inches(7.0))
 elif line.startswith('# '):doc.add_heading(plain(line[2:]),0)
 elif line.startswith('### '):doc.add_heading(plain(line[4:]),2)
 elif line.startswith('## '):doc.add_heading(plain(line[3:]),1)
 elif line.startswith('- '):doc.add_paragraph(plain(line[2:]),style='List Bullet')
 else:doc.add_paragraph(plain(line))
 i+=1
footer=sec.footer.paragraphs[0];footer.text='DriveLab 车辆生成实验 · 10 identities · 2026-10-23 · 源条件诊断，非论文全量基准';footer.style='Caption'
doc.save(O/'汇报正文.docx')
# Plain-text fallback preserves tables as tab-delimited text.
plainlines=[]
for line in lines:
 if line.startswith('|'):
  if re.fullmatch(r'[|\-: ]+',line):continue
  plainlines.append('\t'.join(plain(x.strip()) for x in line.strip('|').split('|')))
 elif line.startswith('!['):plainlines.append(plain(line[1:]))
 else:plainlines.append(plain(re.sub(r'^#+ ','',line)))
(O/'汇报正文.txt').write_text('\n'.join(plainlines),encoding='utf-8')
css='''body{margin:0;background:#eef2f7;color:#17243b;font-family:system-ui,"Microsoft YaHei",sans-serif;line-height:1.75}main{max-width:1160px;margin:auto;background:white;padding:44px}h1{font-size:32px;line-height:1.3}h2{margin-top:38px;border-bottom:2px solid #dae5f3;padding-bottom:9px}h3{margin-top:26px}a{color:#1d4ed8}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}th,td{border:1px solid #dbe3ef;padding:8px;text-align:left}th{background:#eff5ff}img{max-width:100%;height:auto}video{display:block;width:100%;max-height:600px;background:#f1f5f9;margin:16px 0}nav{padding:20px;background:#eaf2ff;border-radius:10px}details{padding:18px;border:1px solid #dbe3ef;margin:16px 0;border-radius:8px}summary{cursor:pointer;font-weight:700;font-size:20px}.source{max-height:230px;width:auto}.links{display:flex;flex-wrap:wrap;gap:18px}.tag{display:inline-block;color:#166534;background:#dcfce7;padding:4px 12px;border-radius:12px}.note{color:#64748b;font-size:14px}@media(max-width:700px){main{padding:20px}table{font-size:11px}th,td{padding:4px}}@media print{body{background:white}main{padding:0;max-width:none}video,nav,details{display:none}h2,h3{break-after:avoid}table,img{break-inside:avoid}a{color:inherit}}'''
body=markdown.markdown(report,extensions=['tables','sane_lists'])
(O/'汇报正文.html').write_text(f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>车辆生成实验汇报</title><style>{css}</style><main><nav><a href="START_HERE.html">返回视频与图片总入口</a> · <a href="汇报正文.docx">Word 可编辑版</a></nav>{body}</main></html>')
cases=json.loads((R/'inputs/random10_cases.json').read_text());cn=['蓝灰 SUV','白色皮卡','银色 MPV','白色 SUV','银色跨界 SUV','深蓝轿车','灰色 MPV','白色轿车','深色轿车','灰色电动 SUV']
start=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>车辆实验 · 导师汇报</title><style>{css}</style><main><p class="tag">10 个对象 · 6 条主路线 · 70 个资产 · 13 个展示视频</p><h1>真实多图与 FLUX 辅助的车辆高斯生成</h1><p>解压后可直接离线浏览，无服务器、无外部脚本、无网络字体。</p><nav class="links"><a href="汇报正文.html">完整汇报与所有表格</a><a href="汇报正文.docx">Word 可编辑版</a><a href="汇报讲稿.txt">三分钟讲稿</a><a href="metrics/summary_all_metrics.csv">主结果 CSV</a><a href="#gallery">十对象全部视频与原图</a></nav><h2>结论与边界</h2><p>综合研究主线建议选 <b>H</b>；原 TRELLIS 工程增量方案选 <b>B0＋4 real</b>。归一化源图局部外观是 <b>R</b> 更好。没有一条路线在所有指标上占优。</p><p class="note">均为十对象 pilot 的源条件指标；不代表留出视角重建或生产验收。详细说明、两套对齐结果和尺寸误差均保留。</p><img src="images/overview/pipeline.png" alt="实际实验流程"><h2>源条件特征指标</h2>{markdown.markdown(main,extensions=['tables'])}<h2>新增像素与感知指标</h2><p>上排：前景归一化。下排：保留相机投影位置与大小。两套口径在计算前固定；误差棒按十个身份重采样。</p><img src="images/overview/pixel_metrics.png" alt="PSNR SSIM LPIPS"><h2>建议展示顺序</h2><p>04 白色 SUV → 02 白色皮卡环绕 → 10 电动 SUV 的身份漂移 → 08 白色轿车的未见面失败案例 → 皮卡精修。十例全部保留，统计没有只选展示例。</p><h2 id="gallery">十对象全部视频、原图与比较</h2><p>七列图片顺序：Source / B0 / R / F / H / B0_real1 / B0_real4。图片为已有对比图原样复制，可直接放入幻灯片。</p>'''
for idx,(c,name) in enumerate(zip(cases,cn),1):
 tag=c['case_id'].replace('random10_','');prefix=f'images/sources/{tag}';comp=f'images/comparisons/{tag}_matched.jpg';video=f'videos/{tag}_six_routes.mp4'
 start+=f'<details {"open" if idx==4 else ""}><summary>{idx:02d} · {name}</summary><p class="links"><a href="{prefix}_full_frame.jpg">完整原始相机 JPEG</a><a href="{prefix}_location.jpg">目标位置标记</a><a href="{prefix}_source_crop.png">未分割原始裁剪</a><a href="{prefix}_source_foreground.png">分割前景</a><a href="images/inputs/{tag}_segmented_flux.jpg">真实与 FLUX 输入</a><a href="{comp}">原尺寸对比图</a><a href="{video}">独立播放视频</a></p><img class="source" src="{prefix}_source_crop.png" alt="真实源图"><img loading="lazy" src="{comp}" alt="七列固定源视角对照"><video controls preload="none" src="{video}"></video></details>'
start+='<h2>白色皮卡追加实验</h2>'
for name,title in [('ablation_count_seed','真实图数量与随机种子'),('trellis_real_assist_seed_comparison','原 TRELLIS 加真实图与随机种子'),('refinement_R_F_comparison','R/F 官方精修前后')]:start+=f'<details><summary>{title}</summary><video controls preload="none" src="videos/pickup_{name}.mp4"></video></details>'
start+='<p class="note">70 个高斯 PLY 与模型权重留在服务器，路径见 reproducibility/asset_paths.json；本包聚焦汇报材料。主实验没有新增场景插入。</p></main></html>'
(O/'START_HERE.html').write_text(start)
script='''本次我们主要比较真实多视图、FLUX 标准化视图以及两者混合，对车辆高斯资产生成的影响。

我们固定了十个新车辆，每辆四张真实训练图、两张留出图，并重新生成两张 FLUX 图。六条主路线包括原 TRELLIS 的 FLUX 双图基准 B0；ReconViaGen 的真实四图 R、FLUX 双图 F、混合六图 H；以及在原 TRELLIS 中加入一张或四张真实图的两条路线。另在白色皮卡上做数量、种子和精修消融，总共七十个资产。

先看白色 SUV 的原图和对比。只用 FLUX 时，前脸和车身细节可能变化；加入真实图后，实例颜色和外观更接近原车。电动 SUV 的例子更明显：FLUX 补出了燃油车式格栅，这个身份漂移会被后续模型继承。环绕视频用于检查三维完整性，不能因为一张图好看就认定三维正确。

定量上，六路线 AOR 都是百分之百，表示全部可加载、十六视角可渲染，不代表全部视觉合格。H 的 DINO-I16 和 ED-R 均值最好；原 TRELLIS 加四张真实图的 CLIP-I16 略高。

新增 PSNR、SSIM 和 LPIPS 后，结论更具体：把前景居中、等比例归一化后，R 三项最好，分别约十六点八一 dB、零点八一四五、零点一四五三。保留相机投影位置后，H 的 PSNR 和 LPIPS 均值更好。排名会随对齐口径改变，所以我们两套都汇报。宽高尺寸误差则是 F 最小，外观指标与几何误差不能互相替代。

综合来看，建议下一阶段以 H 作为研究主线，保留 R 作为强基线；如果要求直接延续原 TRELLIS 工程，就选加四张真实图。这个建议考虑多视角身份保持和整体投影，不意味着 H 每项都赢。R 精修在代表皮卡上有收益，F 精修不一致，因此暂不全量默认启用。

最后说明边界：这些是十个对象的源条件试验，源图可能参与生成，不是留出视角重建精度。本批选对象和相机使用了 Waymo 原生元数据，没有完整重跑 GroundingDINO 和 Qwen；没有未见面的真实 GT，也还没有对这十个对象完成场景插入。下一步重点是扩大对象和种子、可靠留出视角评估、几何比例及场景接地。
'''
(O/'汇报讲稿.txt').write_text(script)
# Copy compact numerical evidence, protocols and source code.
for name in ['random10_metrics.json','random10_metrics.csv','random10_pixel_metrics.json','random10_pixel_metrics.csv','random10_pixel_protocol.json','random10_evaluation_protocol.json','random10_geometry_cost.json','random10_additional_metrics.json','random10_additional_metrics.csv','random10_quality_notes.json','random10_heading_sensitivity.json']:
 shutil.copy2(R/'docs'/name,O/'metrics'/name)
for name in ['evaluate_pixel_metrics.py','evaluate_asset_metrics.py','aggregate_asset_metrics.py','prepare_metric_sources.py','render_metric_views.py']:
 shutil.copy2(R/'scripts'/name,O/'reproducibility'/name)
for name in ['random10_deliverable_audit.json','random10_additional_audit.json','random10_provenance.json','clip_weights.json','dinov3_weights.json']:
 shutil.copy2(R/'docs'/name,O/'reproducibility'/name)
shutil.copy2(R/'docs/random10_asset_index.json',O/'reproducibility/asset_paths.json')
with (O/'metrics/summary_all_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
 rows=[]
 for v in V:
  row=dict(route=v,n_identities=10,AOR_percent=100)
  row.update({k:emb(v)[k]['mean'] for k in ['CLIP_I16','DINO_I16','ED_R']})
  for mode in ['normalized','native']:
   for k in ['PSNR_dB','SSIM','LPIPS_alex','silhouette_IoU','PSNR_union_dB','SSIM_union']:row[mode+'_'+k]=pix(mode,v)[k]['mean']
  row.update({k:geo(v)[k] for k in ['width_MAPE_percent','height_MAPE_percent','inference_seconds_mean','peak_allocated_GiB_mean']});rows.append(row)
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(O/'README.txt').write_text('''导师汇报包（2026-10-23）

1. 完整解压，再双击 START_HERE.html：离线查看图片、视频与指标。
2. 汇报正文.docx：可编辑的 Word 文档，包含流程、实验区别、完整对比表及最终建议。
3. 汇报正文.html：浏览器阅读／打印；汇报正文.txt：纯文本备份。
4. 汇报讲稿.txt：约三分钟讲述提纲。
5. videos/：十对象全部六路线视频＋三个代表对象追加视频，共十三个；保留现有文件，未降画质转码。
6. images/sources/：完整原始相机 JPEG、原始裁剪、白底前景和目标位置标记。
7. images/comparisons/：固定源视角七列对比，原样复制，可直接用于 PPT。
8. metrics/summary_all_metrics.csv：可直接用 Excel 打开的主表；其余 JSON/CSV 保留逐对象数值和口径。
9. reproducibility/：指标实现、权重来源、审计和服务器 PLY 路径。模型权重、环境和 PLY 不随包分发，代码复跑需要原服务器环境，不是独立一键安装包。

PSNR、SSIM、LPIPS 是固定训练源视角诊断；两种对齐口径均保留，不是留出或新视角 GT 重建评估。十三个视频覆盖全部十对象和追加实验，不掩盖失败外观。

SHA256SUMS.txt 可核验包内文件。开始展示建议打开 04 白色 SUV → 02 白色皮卡 → 10 电动 SUV → 08 轿车失败例。
''')
print('REPORT COMPLETE')
