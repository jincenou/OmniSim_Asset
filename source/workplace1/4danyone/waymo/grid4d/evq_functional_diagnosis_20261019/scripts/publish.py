from summarize import *
import datetime,html,hashlib,shutil,ast

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();rows=load(D/'residual_summary.csv');loc=load(D/'evidence_locality.csv');disp=load(D/'displacement_window_summary.csv');keys=sorted(set(r['key'] for r in rows));plan=json.loads((D/'PLAN_DIAGNOSTIC.json').read_text());windows=[];locality=[]
 for key in keys:
  rs={r['mode']:r for r in rows if r['key']==key and r['scope']=='failure_window' and r['roi']=='expanded_0'}
  windows.append(dict(key=key,**{f'{m}_{k}':r[k] for m,r in rs.items() for k in ['RGB_MSE','soft_undercoverage','positive_excess','local_lpips','temporal_RMSE','known_rows','unknown_rows']},position_MSE_relative=rs['M11']['RGB_MSE']/rs['M01']['RGB_MSE']-1,color_MSE_relative=rs['M11']['RGB_MSE']/rs['M10']['RGB_MSE']-1,position_deficit_relative=rs['M11']['soft_undercoverage']/rs['M01']['soft_undercoverage']-1))
  if key.endswith(('R2','R3')):
   w=plan['selections'][key[:3]]['window'];x=[r for r in loc if r['key']==key and r['target_view']==w['view'] and w['start']<=r['frame']<=w['end'] and r['roi']=='expanded_0'];den=sum(r['token_mass'] for r in x);locality.append(dict(key=key,token_mass=den,**{k+'_fraction':sum(r[k] for r in x)/den for k in ['compatible_mass','torso_mass','wrong_side_mass','background_mass','unknown_mass']},read_distance_px=sum(r['final_read_distance_mean']*r['token_mass'] for r in x)/den))
 write(D/'window_effects.csv',windows);write(D/'locality_window_summary.csv',locality)
 # Static observations actually inspected through image tool; no claim of full temporal viewing.
 visual=dict(date_utc=now,reviewer='assistant image inspection',window_contacts={k:dict(view=plan['selections'][k[:3]]['window']['view'],frames=[22,34,46] if k.startswith('N06') else [81,93,105],modes=['REF','M00','M10','M01','M11']) for k in keys},last_frame={'N11_R1':{'views':[0,8],'frame':120},'N11_R2':{'views':[0,8],'frame':120},'N11_R3':{'views':[0,8],'frame':120}},support_overlays=[dict(key=k,frame=f,view=v) for k,f,v in [('N11_R3',93,1),('N06_R2',34,1),('N06_R3',34,1)]],texture=[dict(key='N11_R3',frame=93,view=1,part='left_arm and small torso proxy crop')],full_25_frame_manual_motion_review='unknown;all six videos decoded but only listed frames visually inspected',uninspected_views_frames='unknown',observations=['N06 all modes retain blurred shoes/pants; R3 f34 shoe shape is better than R1/R2 but not restored reference detail','N11 f93 forearm remains shortened across all modes including M00; no output switch restores it','N11 f120 views0/8 front foot remains blurred across all modes; position-on often cleans other scattered regions','N11 support1 f93 source has forearm and garment marks; quarter scale loses fine detail; output lacks forearm and marks','N06 support1 f34 dominant R3 reads cluster near knee; R2 reads are more spread, including opposite leg; not reliable material correspondence'])
 save(D/'visual_review.json',visual)
 decision=dict(created_utc=now,status='mixed_mechanisms_not_separated',new_training_started=False,old_default_replaced=False,A_direct_position_harm=dict(judgment='not stable across fixed cases',evidence='All6 M11 improve window RGB MSE vs M01 by26.20–40.10%;5/6 reduce deficit by20.27–36.00%',counterexample='N06 R3 deficit0.0536047->0.0564145 (+5.24% relative), while excess0.156769->0.071869 and RGB error fall',unknown='Posthoc OOD cannot separate training-history interactions; unseen frames and material geometry unknown'),B_color_value=dict(judgment='local color value, not uniformly beneficial',evidence='M11 vs M10 target-window MSE falls5.33–10.22%;alpha exactly unchanged;N11 reliable torso MSE also falls',counterexample='N06 torso core MSE increases6.89–10.33% with DC on acrossR1/R2/R3',unknown='No true unseen-view ground truth; DC gains not exclusively Query gains'),C_evidence_locality=dict(judgment='coarse read mismatch is a candidate, majority-wrong claim unsupported',evidence=locality,counterexample='N06 R3 compatible proxy mass exceedsR2 and read distance reduces; not all nodes read torso',unknown='75.2–80.0% N06 and93.2–94.2% N11 non-null read mass unknown after visibility/part gates; attention is not causal attribution'),D_representation_bottleneck=dict(judgment='candidate only',evidence='N11 reference texture and forearm visible in support1f93 but output blurred; contribution-weighted Gaussian support measured',counterexample='Input downsampling loses detail and many reads are not demonstrated compatible; multiple small Gaussians and nonlinear decoder can represent high frequency',unknown='Cannot separate read locality, topology, shared deformation and effective texture capacity; no strict low-rank proof'),E_base_problem=dict(judgment='terminal base contains failures under posthoc switches',evidence='N11 forearm f93 and foot f120 remain absent/blurred in M00/M01/M10/M11',counterexample='M00 is jointly-trained base, not independent R0; original native base culpability not established',unknown='training history vsopacity/scale vs motion geometry vs generated multiview inconsistency remain coupled'),single_next_recommendation=dict(action='Do not choose a new parameter-training recipe yet. First disambiguate visibility/material-proxy evidence in the same N11 fixed window using saved observations. If locality mismatch is confirmed, the sole next training factor should be evidence-reading center/granularity, retaining position path and all other settings.',why_not_color_only_now='Stable direct position harm prerequisite absent; current position path largely improves fixed-window coverage.',why_not_other_changes='No separated evidence for offset bound,anchor count,SH order,extra regularizers or a new backend.',conditional_color_only_plan='NEXT_TRAIN_PLAN.md fully specifies requested C1/C2/C3 cold-start test as a candidate, not a currently supported cure or queued job',requires_future_confirmation=True))
 save(D/'decision.json',decision)
 table='|对象|M00 MSE|M10 MSE|M01 MSE|M11 MSE|位置开启：欠覆盖相对变化|颜色开启：MSE相对变化|\n|---|---:|---:|---:|---:|---:|---:|\n'
 for r in windows:table+=f"|{r['key']}|{r['M00_RGB_MSE']:.6f}|{r['M10_RGB_MSE']:.6f}|{r['M01_RGB_MSE']:.6f}|{r['M11_RGB_MSE']:.6f}|{r['position_deficit_relative']*100:+.2f}%|{r['color_MSE_relative']*100:+.2f}%|\n"
 ltable='|对象|读取到最终投影距离 px|相容部位权重|错误侧权重|躯干权重|背景权重|未知权重|\n|---|---:|---:|---:|---:|---:|---:|\n'
 for r in locality:ltable+=f"|{r['key']}|{r['read_distance_px']:.2f}|{100*r['compatible_mass_fraction']:.2f}%|{100*r['wrong_side_mass_fraction']:.2f}%|{100*r['torso_mass_fraction']:.2f}%|{100*r['background_mass_fraction']:.2f}%|{100*r['unknown_mass_fraction']:.2f}%|\n"
 report=f'''# EVQ 固定终态功能分解：诊断完成，未启动训练

实际执行日期：{now}（UTC；北京时间2026-10-19）。N06/N11 × R1/R2/R3六个32999终态已完成四种输出模式、查询位置与支撑尺度诊断。**当前证据不支持把“新增位置残差损伤肢体”作为统一解释，也不足以确认多数查询读错部位。下一轮训练配方暂不发布；请求中的冷启动颜色-only预案已经写好，但不自动执行。**

[视频与同帧图](comparison/index.html) · [固定窗口简表](window_effects.csv) · [完整逐帧指标](residual_modes.csv) · [条件效应及相互作用](residual_interactions.csv) · [五项决策](decision.json) · [下一训练预案](NEXT_TRAIN_PLAN.md)。所有输出均为 diagnostic，不是新训练资产。

## 做了什么、如何核验

每个检查点、每个时刻只使用同一个base几何、SH、opacity、绑定和Z缓存：M00关闭新增位置/DC；M10仅开位置；M01仅开DC；M11两者全开。不重新为各模式生成Z、投影或绑定，不混合不同模型。保留原生7维旋转/平移和原尺度/旋转；新增位移未额外旋转协方差。SH颜色保持CUDA分支（最低值截断），评测灰背景合成后沿用[0,1]截断。

六组共561个含前帧依赖的相机/时刻检查，M11相对已交付raw RGB/alpha最大差**0**；alpha(M00)=alpha(M01)、alpha(M10)=alpha(M11)最大差**0**。原生协方差与预计算协方差光栅化探针最大差9.24e-7，低于1e-5。原渲染器对最终旋转四元数不再归一化，统计亦保持此实际行为。

固定H三视角×13时刻，加N06 v16帧22–46、N11 v00帧81–105；去重并增加前帧只为光流计算。旧核心与扩展ROI、LPIPS裁剪、光流门控公式未变，M11重算与旧对应局部指标最大差小于1e-9。两段左侧扩展窗口各25帧可测；N11全范围原有**20条左侧unknown**另存完整清单，没有由本轮子集抹掉。训练域支持视角仍为1/13/7/19，H从未进入查询输入。

## 位置与颜色作用

下面为固定失败窗口左侧扩展ROI，RGB MSE按有效ROI像素加权；欠覆盖按参考alpha质量加权。位置效应固定颜色开（M11对M01）；颜色效应固定位置开（M11对M10）。百分比为相对变化，未据此设置新通过线。

{table}

位置开启使六组RGB MSE降低26.20%–40.10%，五组欠覆盖降低20.27%–36.00%。反例是N06 R3：欠覆盖0.0536047→0.0564145，增加5.24%；但超额alpha0.156769→0.071869，RGB误差同时下降。它是覆盖/外溢之间的变化，不能单独据一个指标把位置头判为有害。

DC开启使六组目标窗口MSE降低5.33%–10.22%，且alpha完全不变，所以这些颜色收益不是颜色伪造了覆盖率。N11躯干核心也改善；**N06躯干核心反而退化约6.9%–10.3%**，因此颜色也不是对所有部位都有效。R1同样有收益，不能都归于视觉Query。

每个ROI/每帧/固定窗口都保存四种绝对值、已知与未知分母，以及E11−E01、E11−E10、E10−E00、E01−E00和J=E11−E10−E01+E00。J不是训练因果贡献比例。M00仍是与新增分支共同训练过的终态base，**不等于R0**；关闭路径是分布外干预，不直接批准部署开关方案。

## 查询实际读取哪里

通过原生光栅器dummy颜色的梯度求得每个Gaussian对ROI的真实alpha合成贡献，再乘固定top4绑定权重和实际两头token权重。只对dummy颜色求导，没有对模型参数反传更新。贡献和与ROI alpha质量一致性通过预注册1e-4相对容差。绑定使用规范坐标与终态属性，所有时刻索引一致；无逐帧节点切换证据。

{ltable}

分母是非空token读取权重；空token另存，不混入此表。部位判断同时要求Gaussian自身投影与参考部位相容、前景有效、深度接近；不满足的读取归unknown，不能说成错误。注意力权重也不等于通过decoder的因果影响。`other_part_mass`包含躯干/错误侧，是总括字段，不能与子项再相加。

N06 R3的读取距离较R2更短、相容权重更多；N11 R3没有出现相同趋势。已看N11 support1/frame93中，主要采样靠近肩/上臂，参考前臂可见而渲染前臂缩短，属于局部错配候选。但N11约94%的权重仍未知，**不能宣称“多数失败由读错部位导致”**。

1/2和1/4分别是176×320、88×160输入。图中显示实际学习偏移加固定十字后的采样：RGB/alpha的双线性加平均池化足迹严格映射回352×640；Sobel通道还向外扩展一个特征像素，详细边界见research/footprint_bounds.json。部位占比统计针对RGB/alpha直接足迹，未冒充Sobel语义归属。全部32个采样点存于缓存，展示只画贡献最高8个Gaussian所关联的前4个节点，避免视觉拥挤；没有把低贡献节点均等解释。

## 位移、支撑与纹理

[displacement_support.csv](displacement_support.csv)给出每帧每ROI的贡献加权分布，[窗口汇总](displacement_window_summary.csv)给出均值。目标窗口中实际位移约0.0143L–0.0291L，屏幕约5.39–9.99px；实际椭球下Mahalanobis均值约7.14–14.74。epsilon=1e-8L²，病态块按预注册条件记NA。N06/N11主要贡献高斯的长轴sigma约6.69–10.34px / 5.01–5.16px；这是局部投影支撑，不是实际表面大小。薄轴会放大归一化位移，不能仅凭这些数值断言0.03L过大。

饱和描述为任一位移分量超过原界限95%，不是新增质量门槛。alpha增减图/前景背景躯干统计见各comparison目录和results/*/alpha_transfer.csv；二维躯干区域不是材料遮挡层。N11手臂/躯干节点贡献分布重叠约0.28–0.30，N06腿/躯干约0.03–0.05；这说明共享关联程度不同，不能证明共享导致缺失。

参考、半/四分之一输入和M11同相机裁剪已并排保存。N11参考仍有服装花纹和细肢体，四分之一图丢失不少细节，M11更多为平滑色块。读取相容性尚不充分，所以**局部外观表示受限仅是候选**。单Gaussian DC只有三个RGB常数，高阶SH是方向变化；但多个小Gaussian及位置编码decoder仍可表达高频，128节点不能证明严格低秩。描述性频谱保留边缘/明暗影响，未称真实服装纹理频率，未做跨相机伪配准PSNR。详见[texture_scale_notes.md](texture_scale_notes.md)。

## 实际视觉观察与边界

六个25帧视频均完整解码。人工图像检查为每个固定窗口首/中/末（N06 22/34/46、N11 81/93/105），以及N11三个分支在v00/v08的末帧120；其他动态闪烁范围unknown，不冒充逐帧看完。六组同帧图显示N06鞋裤仍模糊，N11第93帧前臂在M00/M01/M10/M11均缩短；第120帧前方脚在四模式都模糊。关位置并没有自动恢复。细节范围见[visual_review.json](visual_review.json)。

## 下一步与执行记录

本轮无法把共同主干/opacity/scale/运动几何/生成多视角冲突各自的贡献分开。**单一建议：先利用已保存N11固定窗口补清部位与可见性对应，再决定下一训练变量。** 如果证实读取粒度错配，下一次只改“锚点级读取到primitive局部读取”，保留位置路径及其他配方；这仅是条件方向，当前不执行。当前不优先跑颜色-only六组，也不扩增offset、锚点、SH或换后端。按任务要求，完整颜色-only冷启动C1/C2/C3预案已写入NEXT_TRAIN_PLAN.md，作为待确认候选，不能称已有因果依据或已排队。

物理GPU0–5并发完成六个只读终态；GPU6/7/0/1并发完成四个视觉读取任务，均由独立tmux服务fdgrid_evqdiag执行。GPU0–7都实际参与；没有为了占卡制造重复训练。CPU媒体第一控制器在子任务完成后未推进，只停止该控制器并在tmux续做未完成媒体，已有文件保留。首次便捷统计因环境无pandas退出，改用已有csv/numpy，无安装；没有改变任何冻结模型或指标。

已读取quality_v3颜色实验：旧固定几何SH拟合C1/C2降低训练误差却损伤H指标，本轮没有重跑。旧C1/C2与本预案的同名分支定义不同。未安装参考论文系统，也未把论文机制未经验证地当成本项目原因。5GiB写入余量保留，不重新施加35GiB限制。旧资产/报告/代码/ROI前后哈希和零更新依据见completion_audit.json。A仍为八人质量参照，旧R3未达到跨困难对象有效性的结论保持不变。
'''
 (D/'FINDINGS.md').write_text(report)
 (D/'texture_scale_notes.md').write_text('''# 纹理与局部支撑：描述性诊断

图像中的REF352是后端使用的352×640生成参考，不是原始Waymo拍摄分辨率。half/quarter是实际平均池化RGB输入，展示用nearest放回同一尺寸；没有超分或增强。六通道还含alpha/Sobel，颜色图不代表全部特征。

真实贡献加权的投影椭圆轴/面积在displacement_support.csv；参考固定部位裁剪的Hann窗、去均值频谱在texture_scale.csv。频率是图像变动尺度，含轮廓、阴影、背景影响，不能当成衣物条纹的语义真值。不同相机的频率与高斯支撑不作强行一一对应；可在同相机图上定性核查。

N11 support1/frame93的前臂及衣服纹理可见，输出仍平滑、缺肢；该区域读图相容性高度未知，不能用“参考有纹理但输出没纹理”单独证明DC表示瓶颈。N06 R3相对R2的读取更靠近腿部，仍有鞋部细节损失，是下一诊断候选而非效果保证。

单Gaussian的DC在其投影内部不随像素变化；高阶SH随观察方向变化。多个小Gaussian叠加与非线性解码可以表达高频。因此不由128锚点推导严格低秩或必然模糊；不据描述性频率设置新通过阈值。

读取分布统计基于真实alpha贡献×固定关联权重×两头平均token权重，最后项只是token合成权重，不是端到端因果贡献。RGB/alpha采样实际有2×2双线性特征格再经过平均池化；Sobel增加邻域，详见research/footprint_bounds.json。图中足迹只展示高贡献节点，完整32token位置保留于diagnostic缓存。
''')
 htmlparts=['<!doctype html><meta charset="utf-8"><title>EVQ 功能诊断</title><style>body{font:16px system-ui;background:#171b21;color:#eee;max-width:1400px;margin:30px auto}a{color:#86d3ff}img,video{max-width:100%}section{border-top:1px solid #555;margin-top:30px;padding-top:20px}summary{cursor:pointer}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #555}</style><h1>EVQ 固定终态：位置/颜色分解</h1><p>只读诊断，未训练。列序：参考、M00原终态base、M10仅新增位置、M01仅新增DC、M11完整。M00≠R0；开关不等于重新训练消融。</p><p><a href="../FINDINGS.md">完整结论</a> · <a href="../window_effects.csv">窗口指标</a> · <a href="../residual_summary.csv">全部汇总</a> · <a href="../NEXT_TRAIN_PLAN.md">待确认训练预案</a></p><p>固定窗口视频每段25帧/25fps，支持0.25倍速。全部自动解码；人工只检查已登记的帧，未声称完整动态质量通过。</p>']
 for key in keys:
  p=D/'comparison'/key;assert (p/'media.json').exists();htmlparts.append(f'<section><h2>{key}</h2><video controls loop preload="metadata" src="{key}/fixed_window.mp4"></video><p><button onclick="this.parentElement.previousElementSibling.playbackRate=.25">0.25倍速</button> <a href="{key}/fixed_window.mp4">下载视频</a></p><img src="{key}/window_contact.jpg"><details><summary>完整25帧同帧页、末帧、alpha差异与读取位置</summary>')
  for path in sorted(p.glob('*')):
   if path.suffix in ['.jpg','.png'] and path.name!='window_contact.jpg':htmlparts.append(f'<p><a href="{key}/{path.name}">{html.escape(path.name)}</a></p>')
  htmlparts.append('</details></section>')
 htmlparts.append('<p>支持图仅用训练视角1/13/7/19。可见性未知不算错误也不算通过；H是历史开发参考，不是真实Waymo新视角真值。</p>');(D/'comparison/index.html').write_text('\n'.join(htmlparts))
 # Capture completion only after weight/ROI/source equality and required files check.
 inv=json.loads((D/'checkpoint_inventory.json').read_text())
 for x in inv:
  x['after']={p:sha(p) for p in x['before']};x['unchanged']=x['after']==x['before'];assert x['unchanged']
 save(D/'checkpoint_inventory.json',inv)
 sources={p:sha(p)==h for p,h in {**plan['source_files'],**plan['roi_hashes']}.items()};assert all(sources.values())
 files=['PLAN_DIAGNOSTIC.json','STATUS.json','checkpoint_inventory.json','residual_modes.csv','residual_interactions.csv','alpha_invariance.json','displacement_support.csv','evidence_locality.csv','texture_scale_notes.md','decision.json','comparison/index.html','FINDINGS.md','NEXT_TRAIN_PLAN.md']
 assert all((D/p).exists() for p in files)
 # No optimizer.step call or backward call exists in new execution code; only dummy autograd.grad.
 calls=[]
 for p in (D/'scripts').glob('*.py'):
  tree=ast.parse(p.read_text())
  for node in ast.walk(tree):
   if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ['step','backward']:calls.append(dict(file=str(p),line=node.lineno,call=node.func.attr))
 assert not calls
 videos=[json.loads((D/'comparison'/k/'media.json').read_text())['files'][0] for k in keys];assert all(v['frames']==25 and v['fully_decoded'] for v in videos)
 audit=dict(state='diagnostic_delivery_complete_with_unknowns',created_utc=now,required_files=files,terminal_assets=6,new_training_assets=0,optimizer_calls=0,zero_updates_evidence='Reachable execution loads portable inference models, all model parameters requires_grad=False; new scripts contain no optimizer.step/backward calls or optimizer construction. Only autograd.grad of dummy colors measures compositing. Not a runtime optimizer profiler.',original_asset_files_unchanged=True,source_and_ROI_unchanged=sources,alpha_replay='alpha_invariance.json',metric_reproduction='research/metric_reproduction.json',six_continuous_videos=videos,manual_scope='visual_review.json',old_N11_left_unknown_rows=20,temporary_feature_cache='cache/* diagnostic only; not registered as trained assets',disk_free_GiB=shutil.disk_usage(D).free/2**30,limitations=['H development bias','single seed','posthoc OOD','support visibility/material correspondence mostly unknown','full temporal manual inspection unknown'],anomalies=[dict(stage='convenience_summary',issue='pandas unavailable',resolution='used existing csv/numpy; no environment changes'),dict(stage='CPU_media_controller',issue='three completed children but controller did not advance',resolution='stopped only own controller PID2887757; tmux async resumed unfinished media; no GPU rerun')],script_sha256={str(p.relative_to(D)):sha(p) for p in (D/'scripts').glob('*.py')})
 save(D/'completion_audit.json',audit)
 status=json.loads((D/'STATUS.json').read_text());status.update(state='complete_readonly_diagnosis',completed_utc=now,training_started=False,new_optimizer_updates=0,decision='mixed_mechanisms_not_separated',next_training='proposal only; requires future review and confirmation',running=[],pending_gpu_jobs=[]);save(D/'STATUS.json',status);print('published',len(windows),'window cases',len(videos),'videos; original hashes unchanged')
if __name__=='__main__':main()
