"""Write evidence-backed delivery and validate required artifacts and links."""
import csv,hashlib,json,os,re,time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
from analyze_query import NEW,ROOT,PLAN,read,csvrows,write
from runtime import save

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()

def main():
    now=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat();media=read(NEW/'comparison/media_validation.json');gate=read(NEW/'gate_results.json');assert not gate['A_confirmation']['triggered'];assert media['new_videos']==27 and media['reused_videos']==6
    assert read(NEW/'research/protocol_audit.json')['state']=='passed';recovery=read(NEW/'research/recovery_audit.json');assert len(recovery['exports'])==len(recovery['independent_replays'])==36
    assert read(NEW/'research/outside_alpha_audit.json')['state']=='passed'
    for family in media['new_families']:
        folder=NEW/'comparison'/family['name']
        for item in family['videos']:assert sha(folder/item['path'])==item['sha256']
    for item in media['reused_families']:
        for video in item['validation']['videos']:assert sha(Path(item['folder'])/video['path'])==video['sha256']
    pictures=[NEW/'comparison'/n/'parts_rgb.jpg' for n in ['N04_query','N04_reference','N06_query','N06_reference','N11_query','N11_reference','N02_alpha','N03_alpha','N08_alpha']]+[ROOT/'optimization_quality_v3_20261018/comparison'/f'{l}_A_fusion/parts_rgb.jpg' for l in ['N01','N07']]+[NEW/'comparison/N11_query/view00_contact.jpg',NEW/'comparison/N04_query/view08_contact.jpg']
    notes={
      'N01':'复用局部图：alpha/A 接近，头发/脸和衣服文字模糊仍在；没有声称全部运动范围已看完。',
      'N02':'alpha/A 固定四部位接近，手和衣服格纹仍模糊；双腿无可靠核心，图中明确写 unknown，不计为零误差或通过。',
      'N03':'alpha/A 固定局部接近，手部拖影与边缘杂色仍存在；不等于所有转身帧已恢复。',
      'N04':'六部位图及 v08 首/中/末：F/U/P/E/Q 接近，鞋/手脸模糊；末帧脚部弥散各短分支仍有。在已看范围没有清楚的 Q 新增大损伤，完整时序视觉判据仍 unknown。',
      'N06':'六部位固定图：v16 f27、v08 f27 的短分支均有明显腿脚缺失/弥散，Q 没有明显修复；alpha/A 保留更多鞋脚。所有后端在选定侧脸仍严重失真。',
      'N07':'复用六部位图：alpha/A 接近但手脸细节仍失真；全局 LPIPS 实际超保护线，未因观感接近改判。',
      'N08':'alpha/A 固定六部位接近，面部、手和鞋仍模糊；没有全时序质量保证。',
      'N11':'六部位和 v00 首/中/末：Q 与 U/P/E 都未修复首帧横伸手臂淡化/缩短；衣服花纹损失和手臂洋红污染仍在。alpha/A 也保留明显颜色错误。'}
    save(NEW/'visual_review.json',dict(state='sampled_visual_review_complete',reviewed_at=now,reviewer='Codex image inspection',exact_images=[dict(path=str(p),sha256=sha(p)) for p in pictures],observations=notes,automatic_decode= dict(new_videos=27,reused_hash_verified_videos=6,validation=str(NEW/'comparison/media_validation.json')),unchecked='All-frame human temporal/texture flicker review remains unknown; decoded frames are not claimed manually watched. No visual pass inferred from numerical pass.',N04_visual_gate='unknown_full_timeline; no obvious new large damage in inspected fixed cores and v08 first/mid/last',alpha_stage_quality='Fixed stage comparison only; no per-person best checkpoint selection'))
    (NEW/'PIPELINE_DECISION.md').write_text('''# 固定流程决策
2026-10-19（北京时间）。Waymo 人体轨迹 → 冻结的 4DAnyone Turbo 多视角 RGB/RGBA/相机 → Grid4D 逐人体优化 → 动态资产与独立回放。

|角色|方案|本轮结论|
|---|---|---|
|质量参照|完整 A，52999 实际更新|继续保留；平均前景/感知最好，但局部外观缺陷及轮廓外残留仍有|
|短预算研究基座|fused，32999|继续用于有限预算机制验证，不是已证明可统一替代 A 的交付方案|
|历史跨后端基线|HUST grid61|保留；轮廓外 alpha 更低，不能只看 PSNR 将其判定全面落后|
|中间质量候选|alpha47999|五人通过旧数值线；N02未知、N07/N11失败，不宣布统一早停|
|Query-0 候选|F+Q，34999|本轮实际优化并导出，但不升为默认配方|

Q 相对同预算 U：三人均通过数值保护线，但 N06/N11 的目标覆盖提升都未达 +0.02；相对 P/E 也未达 +0.005。不能宣称查询独立收益成立。Q 相对 A 的全部旧保护线三人都没有通过。A+U/Q 四条确认未触发，其收益未知，不伪造为零或失败。

Q 只是训练域部位—时间需求驱动的监督选择；在选中观测自己的相机/时间运行原损失，不搬运其他时刻颜色，不新增可学习 attention/Transformer。没有逆概率校正，改变了期望监督分布，不能称原均匀目标的无偏加速。

F+Q 相对 A 的更新数少约 33.96%，不是运行时间同比节省。已有缓存起点上的一次增量实验只支付加载、需求/优化/保存/评测，不能重复宣传省了前缀 20000 次更新。共享 GPU 墙钟只作实测记录。

动态资产仍需匹配 runtime、hash/deform 和相机时间元数据。没有真实 Waymo 不可见面真值、世界尺度/地面接触保证或任意新动作驱动。下一步仅见 [NEXT_STEP.md](NEXT_STEP.md)。
''')
    (NEW/'FINDINGS.md').write_text('''# Query-0 与八人 alpha/A：实验记录
2026-10-19（北京时间）。本轮完成三人 × 四组的 12 条真实短优化，共 24000 次新增参数更新，36 个阶段快照已新进程回放；八人 alpha47999 部位/时间评价补齐。**Query-0 未达到预注册增益条件，未触发 A 起点的四条确认。** 不新增八人重训，不替换已有配方。

入口：[指标与均值](METRICS.md) · [完整对照视频/局部图](comparison/index.html) · [12 个终态资产与 36 个快照](ASSETS.json) · [逐条判据](gate_results.json) · [实际视觉检查范围](visual_review.json)。

## 本轮做了什么
N04 检查副作用，N06 检查运动腿部，N11 检查运动手臂。每人 U/P/E/Q 从同一个 fused32999 完整状态恢复，保持高斯点数、SH 阶数、可训练参数、损失和学习率一致，各训练 2000 次，到34999。实际保存点为33499/33999/34999；500/1000快照仅作曲线，不选best。需求表只用21训练视角的168固定观测，起点共享评分、1000次更新后各刷新一次；H={0,8,16}不进入训练或需求排名。

U 保持普通 stack 采样；P 对可信部位/时间均衡；E 对全前景误差优先；Q 对部位/时间欠覆盖与 RGB 残差优先。偶数更新1000个观测严格配对；正则扰动 RNG、实际LR数组、恢复点也配对。独立选择器的完整2000项序列已重放，500/1000保存点恢复后的下一选择已验证。无候选时使用已登记均匀支路，本轮没有因此引入H。

## 收益有多大
三人 Q 相对同预算 U：前景 PSNR +0.011623dB，SSIM -0.000132，LPIPS +0.000314（更差）。相对未续训 F 的前景PSNR +0.057104dB，大部分同样可由U普通续训达到，不能全算查询收益。

|目标核心|U覆盖|P覆盖|E覆盖|Q覆盖|Q−U，百分点|要求|
|---|---:|---:|---:|---:|---:|---|
|N06左腿|0.953229|0.953506|0.952993|0.953880|+0.0651|至少+2，失败|
|N11左臂|0.799493|0.798724|0.799686|0.801066|+0.1573|至少+2，失败|

Q 相对 P/E 的较差侧覆盖收益也低于0.5个百分点，未达到查询超越简单选择的要求。Q三人相对U的PSNR/SSIM/LPIPS、全部可信部位覆盖、运动补偿RMSE及轮廓外alpha保护线均通过数值检查，但N04全时序视觉范围未知，数值通过不是质量交付承诺。Q三人相对A全部旧保护线均未通过。

目标 RGB MSE 同样保留：N06左腿 U=0.0192766、Q=0.0191556；N11左臂 U=0.0230946、Q=0.0230469，改善很小。N11 v00左臂最差25帧窗口81–105的覆盖 U=0.532213、Q=0.533994、A=0.621576，23个时刻有可信核心；缺少的2个时刻不当成零误差。所有六部位完整曲线、有效光流比例和最差窗口见 [part_time_metrics.csv](part_time_metrics.csv)、[part_summary.csv](part_summary.csv)、[worst25.csv](worst25.csv)。窗口由各方法自己的最差值诊断选择，不用于选终态。

## 为什么变化很小：已证实与未知
选择器确实生效。N06 的P左右腿优先选择169/175次，Q为178/183次，改变很弱；初始48格需求分布的归一化熵0.99872，接近均匀。N11左臂从P158次增至Q217次，需求确实更偏向困难手臂，但覆盖仍只微增。前者说明当前归一化/混合配方产生的监督重分配有限，后者说明增加曝光本身不足以解决这个固定容量短分支的问题。

没有证据表明是选择器没执行或整体点数失活造成的：点数固定，N11有效opacity点从10816到Q10814，N06从16962到16956，没有大幅整体失活。但投影中心数量不等于可靠表面归属，不能据此排除局部几何支撑不足。评分只代表对生成参考的残差，无法分离生成不一致、形变/几何容量、损失耦合与优化预算各自的因果贡献。

已看固定图中，N06短分支腿脚缺失/弥散、N11首帧横伸手臂缩短/变淡都仍在。N11花纹和肤色污染仍未解决；alpha/A也保留明显颜色错误。保守核心避开模糊边界、遮挡及低置信区域，因此95%的腿核心覆盖不等于整条腿和脚都完整，这也解释了高覆盖与明显脚缺失可以并存。未将该观察编造为完整语义真值评价。

## alpha47999 与 A52999
八人固定阶段比较，无新增优化。N01/N03/N04/N06/N08通过旧全部数值线；N02没有已测指标失败，但双腿ROI/运动补偿指标未知，因此不判通过；N07的LPIPS失败，N11的SSIM/LPIPS失败。平均alpha前景PSNR=23.782856，A=23.817128，差0.034272dB；平均LPIPS差+0.001738。均值接近不能抵消特定人物的失败，不能据此发布统一48k早停。完整表在 [alpha48_evaluation.csv](alpha48_evaluation.csv)、[alpha48_gates.json](alpha48_gates.json)。

八人平均包含备用N11；主七人和N11单独统计也保存在cohort_metrics.csv。参考全部是生成H开发观测，不是真实Waymo未观测视角，不报告原论文无法复现的真实几何指标。

## 成本与资产
12条2000步训练子进程各87.87–165.85秒（约1.46–2.76分钟），含加载/保存及共同诊断刷新，不含前端或父检查点构建；不同GPU共享负载，不能作为四种方法的公平速度排序。每条4000次训练渲染、2000次backward、2000次optimizer更新；1000步刷新额外336次无梯度渲染。起点168观测评分每人只计一次，U/P产品化本不需要诊断开销。加载、循环、保存、评分、回放、评测分别见 [costs.csv](costs.csv)、[cost_summary.json](cost_summary.json)。已有同设备只读探针直接复用，不再为了计时重训。

36个新快照均通过独立新进程回放，禁止读取原Random8数据，包含12.25非整数时间及黑/灰/白背景；PLY与恢复张量、hash/deform逐项一致，优化器步数增量正确。八个alpha资产也均有相同无源图回放证据，其中N04/N11本轮补强了旧检查的读文件隔离记录。资产是PLY+网络+元数据+匹配runtime，不是单独PLY即可自由换动作。

## 已保留的执行异常
alpha补评初次缺少私有hash扩展路径，已用正确环境修复；需求诊断初次误把原生7维旋转/平移当作3维加法，已改为读取原生renderer返回的world_means。二者均在首次新参数更新之前修复，原失败状态/日志保留，冻结的训练代码未事后调整。

交付阶段旧N04/N11回放报告字段与新规范不同，初次汇总明确退出；没有补一个假字段，而是在只读旧资产的新进程中补做隔离回放。CPU媒体/alpha汇总线程控制器没有继续推进，停止的仅为本轮两个控制器，保留已完成媒体，改为async有界子进程并正常结束。没有因此改变训练、参数、评分或评价图像。

## 决策与唯一下一步
继续以A作质量参照、F作短预算研究基座、HUST作跨模型基线；不升级Q为默认，不追加A确认或参数扫描。唯一建议是复用N06/N11的已保存错误窗口，做局部几何投影支撑与生成跨视角一致性的联合定位，先判断缺失由何处产生，再决定是否需要容量/表示改动；这一步没有在本轮偷偷启动新训练，详见 [NEXT_STEP.md](NEXT_STEP.md)。
''')
    (NEW/'NEXT_STEP.md').write_text('''# 唯一下一步：定位困难肢体的支撑与监督冲突
本轮不追加训练。保留完整A质量参照和F研究基座；Q-0没有达到预注册增益，不推广到八人，不换seed/阈值重试。

下一步建议只对N06运动腿部与N11横伸手臂的已保存错误窗口，联合检查原生形变后的可见高斯支撑和生成多视角参考的一致性：区分“原生观测有清楚肢体、后端缺失”与“相机/时间间生成参考相互矛盾”。依据现有需求表的支持统计、raw浮点预测、固定核心及完整视频展开，不重复颜色搬运，也不把投影中心直接当材料点。曝光增加而覆盖几乎不变、有效opacity总数未明显下降，是进行这一定位的证据；它们尚不能单独证明容量不足。

先形成一份可指向具体帧/相机/部位的因果候选诊断，再决定是否做表示/容量改动。不要在定位之前自动开启一轮大规模loss或新后端扫描。
''')
    (NEW/'README_run.md').write_text('''# 本轮执行与查看
根目录 `/mnt/ojc/workplace1/4danyone/waymo/grid4d/query0_fused`；环境 `/mnt/ojc/miniconda3/envs/grid4d`。先看 [FINDINGS.md](FINDINGS.md)、[METRICS.md](METRICS.md)、[comparison/index.html](comparison/index.html) 和 [ASSETS.json](ASSETS.json)。

训练实际入口为 `bash scripts/run.sh query_queue`，由独立 tmux 服务 `fdgrid_query0` 运行12个分支。已全部完成，不要再执行一次覆盖输出。修复后的alpha补评使用 `bash scripts/run.sh alpha_queue --repair`（准确参数与当时阶段命令以 EVALUATION_REPAIR_STATUS.json 为准）；补充两个只读回放入口是 `bash scripts/run.sh alpha_replay_queue`。任务退出码和每条实际参数可在队列源码、状态和日志对应核查。

CPU汇总实际命令：
```bash
cd /mnt/ojc/workplace1/4danyone/waymo/grid4d/query0_fused
/mnt/ojc/miniconda3/envs/grid4d/bin/python scripts/analyze_query.py
/mnt/ojc/miniconda3/envs/grid4d/bin/python scripts/summarize_query.py
/mnt/ojc/miniconda3/envs/grid4d/bin/python scripts/audit_query.py
```

完整媒体与alpha残留统计最后由 `fdgrid_query0_delivery` 独立tmux的 `media`、`alpha_excess` 会话运行 `scripts/publish_query.py`、`scripts/alpha_excess.py`；任务完成后会话自然退出。只发布完整比较组，不生成占位/半成品对照，N01/N07复用既有媒体。

新终态示例 `runs/N06_fused_Q`：`point_cloud/iteration_34999/point_cloud.ply`、`deform/iteration_34999/deform.pth`、`config.json`、`asset_camera_data.json` 是动态渲染所需资产；`recoveries/34999.pt` 是完整优化恢复。查询器只参与训练，回放不读取原始图像。实际新进程回放命令示例（会重写同名本轮验证报告，非必要不要重复执行）：
```bash
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=GPU-cd4b2342-fba4-2b42-beb9-60b698debc2d
/mnt/ojc/miniconda3/envs/grid4d/bin/python scripts/validate_quality.py --run "$PWD/runs/N06_fused_Q" --iteration 34999
```

若要开发恢复入口，除Gaussians/hash/deform/Adam/RNG外，必须恢复 `query_selector`、`base_sampler_state`、基础采样序列、剩余stack和当前demand_table。旧通用训练器不会自动恢复这些新字段，不能静默用旧入口替代。已验证500/1000状态的选择器及基础采样下一项完全一致；未承诺CUDA完整优化逐位可复现。

每条新资产保留500/1000/2000三个阶段，这些是本轮规定的曲线和完整恢复证据，不是随意多余输出。原始数据、旧成果、环境和扩展没有修改；所有新增文件在本目录内。
''')
    # Preserve original failed attempts and training timings, then publish final composite status.
    for src,dst in [('STATUS.json','research/TRAINING_STATUS.json'),('EVALUATION_STATUS.json','research/EVALUATION_INITIAL_STATUS.json')]:
        if not (NEW/dst).exists():(NEW/dst).write_bytes((NEW/src).read_bytes())
    training=read(NEW/'research/TRAINING_STATUS.json');assert len(training['jobs'])==12 and all(j['state']=='complete' for j in training['jobs'].values())
    evaluation=dict(state='complete_with_preserved_repaired_attempts',completed_at=now,alpha_people=8,query_terminal_people=3,query_terminal_assets=12,query_snapshot_replays=36,old_alpha_new_guarded_replays=2,initial_failed_attempts='research/EVALUATION_INITIAL_STATUS.json',repaired_alpha='EVALUATION_REPAIR_STATUS.json',extra_replay='research/ALPHA_REPLAY_STATUS.json',denominator_audit='research/aggregation.json',outside_alpha_audit='research/outside_alpha_audit.json',media_validation='comparison/media_validation.json',visual_review='visual_review.json');save(NEW/'EVALUATION_STATUS.json',evaluation)
    save(NEW/'STATUS.json',dict(state='complete',completed_at=now,initial_training_started_at=training['started_at'],initial_training_ended_at=training['ended_at'],jobs=training['jobs'],alpha_evaluation='complete',conditional_A_confirmation='not_triggered_numeric_target_gates_failed',pending_training=0,pending_evaluation=0,pending_delivery=0,quality='Query-0 not accepted as improvement; complete execution is not quality success',next_step='NEXT_STEP.md proposal only; no additional training launched'))
    (NEW/'REPORT.md').write_text('# Query-0 当前交付\n\n2026-10-19（北京时间），本轮执行完成，Query 增益未过预注册条件。\n\n- [实验记录与结论](FINDINGS.md)\n- [Query三人及八人平均指标](METRICS.md)\n- [完整视频/局部图](comparison/index.html)\n- [资产](ASSETS.json) · [判据](gate_results.json)\n- [流程决策](PIPELINE_DECISION.md) · [下一步](NEXT_STEP.md)\n- [执行与恢复](README_run.md) · [完成核验](completion_audit.json)\n')
    deliver=['PLAN.json','ANCHOR_AUDIT.json','STATUS.json','EVALUATION_STATUS.json','alpha48_evaluation.csv','alpha48_gates.json','query0_definition.md','candidate_manifest.json','paired_metrics.csv','part_time_metrics.csv','gate_results.json','costs.csv','ASSETS.json','README_run.md','comparison/index.html','PIPELINE_DECISION.md','FINDINGS.md','NEXT_STEP.md','METRICS.md','cohort_metrics.csv','stage_metrics.csv','part_summary.csv','worst25.csv','outside_alpha.csv','visual_review.json','research/protocol_audit.json','research/recovery_audit.json','cost_summary.json']
    assert len(list((NEW/'sampling_traces').glob('*.jsonl')))==12 and len(list((NEW/'demand_tables').glob('*.json')))==15
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['src','href'])
    links=Links();links.feed((NEW/'comparison/index.html').read_text())
    for url in links.links:
        if not urlsplit(url).scheme:assert (NEW/'comparison'/unquote(url.split('#')[0])).exists(),url
    artifacts=[]
    for p in deliver:
        path=NEW/p;assert path.is_file() and path.stat().st_size>0
        if path.suffix=='.json':json.loads(path.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
        artifacts.append(dict(path=p,bytes=path.stat().st_size,sha256=sha(path)))
    save(NEW/'completion_audit.json',dict(state='complete',completed_at=now,protocol='research/task.txt',checks=[dict(requirement=r,evidence=e,state='passed') for r,e in [
      ('Frozen inputs/recipe/parent optimizer state and train-only demand','research/protocol_audit.json'),
      ('12 real 2000-update continuations,36 exported snapshots and independent replay','research/recovery_audit.json'),
      ('Eight fixed alpha/A full timeline parts/flow comparison,unknown retained','alpha48_gates.json'),
      ('Fixed terminal U/P/E/Q comparisons and conditional A gate','gate_results.json'),
      ('All six parts,RGB/coverage/flow,worst25 and outside/excess alpha','part_time_metrics.csv;worst25.csv;outside_alpha.csv'),
      ('Measured timings,RNG/selection traces,exposure and demand diagnostics','costs.csv;research/sampling_behavior.json'),
      ('27 new fully decoded videos plus6 matching reused videos;local gallery links exist','comparison/media_validation.json'),
      ('Human visual inspection scope explicitly bounded,unknown not pass','visual_review.json'),
      ('Quality decision,next step,asset contract and failed attempts retained','PIPELINE_DECISION.md;NEXT_STEP.md;README_run.md;EVALUATION_STATUS.json')]],artifacts=artifacts,conditional_A_runs=0,quality_success=False,claim='Requested execution and reporting complete; Query effectiveness not established; no statistical significance or real Waymo novel-view truth claim'))
    print('Delivery complete',now,flush=True)
if __name__=='__main__':main()
