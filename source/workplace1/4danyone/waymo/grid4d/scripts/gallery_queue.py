import asyncio
import json
import os
from baseline_queue import ROOT, save, now

LABELS = ['N01', 'N02', 'N03', 'N04', 'N06', 'N07', 'N08', 'N11']


def index():
    sections = []
    for label in LABELS:
        folder = ROOT / 'gallery' / label
        assert json.loads((folder / 'complete.json').read_text())['state'] == 'complete'
        selected = json.loads((folder / 'selection.json').read_text())['worst']
        dynamic = f'<p><a href="../results/{label}/33000/view00.mp4">完整五联视频 v00</a> · <a href="../results/{label}/33000/view08.mp4">v08</a> · <a href="../results/{label}/33000/view16.mp4">v16</a></p>' if label in ['N04', 'N11'] else ''
        sections.append(f'''<section id="{label}"><h2>{label} {'备用困难例，单列' if label == 'N11' else '主样本'}</h2>
<p>最差片段：{selected['part']}，视角 {selected['view']}，帧 {selected['start']}–{selected['end']}。窗口由四个历史后端的局部均方误差共同选出；不用于选择训练检查点。</p>
{dynamic}<video controls loop preload="none" src="{label}/worst_segment.mp4"></video>
<p>上方为完整的一秒诊断片段。每行一个后端，四列为生成参考 / RGB / alpha / 绝对误差。人物缺失也计入误差。</p>
<details><summary>六部位局部对照：RGB、alpha、误差</summary><a href="{label}/parts.jpg"><img loading="lazy" src="{label}/parts.jpg"></a></details>
<details><summary>部位 × 时间热图</summary><a href="{label}/time_metrics.png"><img loading="lazy" src="{label}/time_metrics.png"></a><p>覆盖越高越好；RGB与运动补偿残差越低越好。灰色为unknown。残差低可能伴随部位消失，须结合覆盖率阅读。</p></details>
<details><summary>来源、前处理、条件与生成参考</summary><a href="../audit/stages/{label}.jpg"><img loading="lazy" src="../audit/stages/{label}.jpg"></a><p>真实源裁剪与生成相机未配准；固定首/中/末时刻，仅作来源检查，不计算二者像素精度。</p></details>
<details><summary>参考定义的六部位核心区域与未知区域</summary><a href="../audit/roi/{label}/review.png"><img loading="lazy" src="../audit/roi/{label}/review.png"></a><p>这是估计的可见核心区域，非完整语义真值。头发、附件、遮挡和定位不可靠区域不强行分配。</p></details>
<p><a href="{label}/selection.json">选片规则、时间和固定裁剪</a> · <a href="../audit/support/{label}/4dgs_grid61/parts.csv">HUST终态投影支撑CSV</a></p></section>''')
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>4DAnyone 后端诊断</title>
<style>body{font:16px/1.65 system-ui,sans-serif;max-width:1250px;margin:32px auto;padding:0 18px;color:#172331;background:#f5f7fa}a{color:#145fa8}nav{position:sticky;top:0;background:#fff;padding:12px;border:1px solid #dce2ea}nav a{margin-right:20px}section{background:#fff;margin:24px 0;padding:22px;border:1px solid #dce2ea;border-radius:8px}video{max-width:100%;max-height:900px;background:#222}img{max-width:100%;height:auto}details{margin:16px 0}summary{cursor:pointer;font-weight:600}.note{background:#fff4d8;padding:16px}</style>
<h1>4DAnyone → 动态高斯：部位与时间诊断</h1>
<p>历史8人×4后端，以及N04/N11新增Grid4D。主样本n=7，备用N11单列。所有比较拟合冻结的生成参考，是开发集诊断，非真实Waymo新视角真值。</p>
<p class="note">本页全部对照均完成后发布。不要把“更平滑”直接当作更好：需同时看局部外观、alpha覆盖和完整动作。低置信的参考区域保留unknown；帧数不是独立人物样本量。</p>
<p><a href="../PROGRESS_LATEST.md">最新进度报告</a> · <a href="../audit/paired_summary.csv">逐人/主7人部位汇总CSV</a> · <a href="../audit/stage_part_audit.jsonl">阶段缺陷证据</a> · <a href="../mechanism/static/results/comparison.png">四个单时刻静态诊断</a> · <a href="../../random8/r20261015_113332/gallery/index.html">历史完整动态对照</a></p>
<nav>''' + ''.join(f'<a href="#{label}">{label}</a>' for label in LABELS) + '</nav>' + ''.join(sections) + '</html>'
    path = ROOT / 'gallery/index.html'; assert not path.exists(); path.write_text(html)


async def main():
    path = ROOT / 'audit/gallery_queue.json'; assert not path.exists()
    state = dict(state='running', jobs={}, started_at=now(), pid=os.getpid()); save(path, state)
    sem = asyncio.Semaphore(2)
    async def job(label):
        async with sem:
            command = ['/mnt/ojc/miniconda3/envs/grid4d/bin/python', '-u', str(ROOT / 'scripts/diagnostic_gallery.py'), '--label', label]
            log = ROOT / 'logs' / f'gallery_{label}.log'
            with log.open('x') as out:
                p = await asyncio.create_subprocess_exec(*command, cwd=ROOT, stdout=out, stderr=asyncio.subprocess.STDOUT)
                record = dict(state='running', pid=p.pid, log=str(log), command=command, started_at=now()); state['jobs'][label] = record; save(path, state)
                code = await p.wait()
            record.update(state='complete' if code == 0 else 'failed', exit_code=code, ended_at=now()); save(path, state)
    await asyncio.gather(*(job(label) for label in LABELS))
    if all(v['state'] == 'complete' for v in state['jobs'].values()):
        index(); state['state'] = 'complete'
    else: state['state'] = 'partial_requires_review'
    state['ended_at'] = now(); save(path, state)


if __name__ == '__main__': asyncio.run(main())
