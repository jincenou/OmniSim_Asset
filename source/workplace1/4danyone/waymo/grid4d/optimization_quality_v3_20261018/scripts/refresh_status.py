"""Small local status summary; terminal training is separate from quality acceptance."""
import asyncio,json,csv,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
NEW=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text()) if p.exists() else None
def put(p,d):
    q=p.with_suffix('.writing');q.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n');q.replace(p)
def refresh():
    now=datetime.now(timezone(timedelta(hours=8))).isoformat();jobs={};rows=[]
    names=['N01_33k','N07_33k','N01_alpha','N07_alpha','N01_A','N07_A','N01_fused','N07_fused','N04_Bmatch','N11_C1','N11_C2']
    for name in names:
        run=NEW/'runs'/name;done=read(run/'complete.json');p=read(run/'progress.json')
        if done:
            state='训练完成，回放/评测/质量结论另查';counter=done.get('total_actual_updates',done.get('actual_updates',done.get('base_grid4d_actual_updates',done.get('counts',{}).get('optimizer_calls'))));jobs[name]=dict(state='training_complete',summary=done)
            if 'color_pass' in done:counter=f"颜色{done['color_pass']}轮 / {done['costs']['color_updates']}次颜色更新"
        elif p:
            state='训练中';counter=p.get('actual_updates',p.get('global_actual_updates',p.get('iteration',p.get('step',p.get('color_pass')))));jobs[name]=dict(state='training',progress=p)
        else:state='等待前置阶段';counter='—';jobs[name]=dict(state='pending')
        rows.append(f'| {name} | {state} | {counter} |')
    ev=read(NEW/'EVALUATION_STATUS.json') or {};repair=read(NEW/'EVALUATION_REPAIR_STATUS.json') or {};counts={}
    for key,v in ev.get('jobs',{}).items():
        effective=repair.get('jobs',{}).get(key,v)['state'];counts[effective]=counts.get(effective,0)+1
    calibrations={label:read(NEW/'research/update_calibration'/f'{label}.json') for label in ['N04','N11']}
    put(NEW/'update_calibration.json',dict(state='complete',labels=calibrations))
    training_terminal=all((NEW/'runs'/n/'complete.json').exists() for n in names)
    evaluation_terminal=ev.get('state')=='jobs_terminal' and repair.get('state')=='jobs_terminal'
    put(NEW/'STATUS.json',dict(state='training_and_evaluation_terminal_review_pending' if training_terminal and evaluation_terminal else 'background_experiments_running',at=now,jobs=jobs,evaluation_counts=counts,training_terminal=training_terminal,evaluation_terminal=evaluation_terminal,quality_accepted=False,original_plan='PLAN.json',fusion_addendum='FUSION_PLAN.json',tmux_server='fdgrid_quality',gpu_authorized=list(range(8))))
    report=f'''# Grid4D A 质量与收敛效率实验：当前进度

更新时间：{now}。本轮仍是 **Waymo → 冻结的 4DAnyone Turbo 视频 → Grid4D 动态资产**。下面“训练完成”不代表质量验收通过。

| 运行 | 状态 | 更新/阶段计数 |
| --- | --- | --- |
'''+ '\n'.join(rows)+f'''

GPU 0–7 已获授权，可共享现有任务的显存；未终止其他项目进程。tmux 服务为 `fdgrid_quality`。已挂载完整 A 续训、融合、条件边界训练、颜色训练及独立回放/评测队列。颜色和边界训练完成后 GPU 用于评测；某卡没有本项目训练不代表任务丢失。

完整 A：32999 次原始更新 → 15000 次 alpha → 5000 次 A，共 52999。融合实验：在原流程 15k 停止增密之后，15k–18k 将 alpha 权重从 0 渐增至 0.1，沿原学习率继续到 32999 实际更新。它省去单独的 15k+5k 阶段，但是否保住质量、是否节省墙钟时间尚待固定终态评测。参见 [融合研究](research/FUSION_RESEARCH.md) 和 [冻结补充协议](FUSION_PLAN.json)。这两个新融合支线来自用户后续要求，独立于原 quality_v3 六分支上限，原 A 对照未更改。

N04 校准选择 λ=0.008670829968553782，三组实际 Adam 附加更新比值约 1.087 / 1.099 / 0.844，触发唯一 5k B-match。N11 的预注册最小目标值候选有一组比值 0.794，低于 0.8，未触发训练。另一个已测系数满足幅度门槛，但不能临时改变已冻结的选择规则；不能声称没有任何系数能匹配。详见 [实际更新校准](update_calibration.json)。

颜色 C1/C2 均从 N11 A 的原始 SH 热启动，只改变颜色。预检已验证零更新复现、实际 CUDA SH 方向、常数颜色累计贡献和微小系数梯度；无几何/动作更新。训练池 MSE 是拟合诊断，不能代替 H 上的 PSNR、LPIPS、部位覆盖或视觉验收。

当前评测状态（已合并排序问题的修复重试）：`{json.dumps(counts,ensure_ascii=False)}`。H 固定为 0/8/16 三视角、13 时刻；终态还评测完整时间轴的固定部位和时间一致性。早期快照仅输出数据，不制作未齐列比较视频。

评测初次把历史按帧排列的记录与按视角排列的记录直接比较，误报相同样本不一致；已改为排序后逐项核对视角/帧/裁剪/输入 SHA256，保留失败日志，只重做受影响评测。没有修改指标公式、参考图、门槛或训练资产。

## 文件入口与后续任务

- [当前机器可读状态](STATUS.json)、[原任务协议](PLAN.json)、[补充融合协议](FUSION_PLAN.json)。
- `runs/`：原始资产、匹配的形变/hash、相机、配置、恢复状态；PLY 单独不构成完整动态资产。
- `results/<运行名>_<快照>/metrics.json`：完成的固定快照指标；终态目录另含 `parts.csv`、`regions.csv`、`replay.mp4`。
- `comparison/`：待全部对照列与评测完整后生成正式拼图和视频，当前不承诺已有完整比较成品。
- `logs/`：每个任务独立日志；`EVALUATION_STATUS.json` 保留原失败，`EVALUATION_REPAIR_STATUS.json` 记录修复后结果。

下一步：完成 A 与融合终态；核验颜色/边界的固定参数和独立动态回放；汇总完整质量—更新次数—耗时曲线及部位/运动指标；对完整对照做视觉复核，再决定是否采用融合流程。当前没有宣布新候选合格，也不按 H 挑最优快照。所有指标针对冻结生成参考，不能解释为真实 Waymo 新视角重建精度。
'''
    (NEW/'REPORT.md').write_text(report)
    metrics=[]
    for path in sorted((NEW/'results').glob('*/metrics.json')):
        d=read(path);metrics.append(dict(method=d['method'],label=d['label'],actual_optimizer_updates=d['actual_optimizer_updates'],color_pass=d.get('color_pass'),**d['mean']))
    if metrics:
        with (NEW/'paired_metrics_partial.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    return training_terminal and evaluation_terminal
async def main():
    while not refresh():await asyncio.sleep(45)
if __name__=='__main__':asyncio.run(main())
