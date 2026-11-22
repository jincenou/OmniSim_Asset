import asyncio
import json
import os
import time
from pathlib import Path
from run_condmdi_cases import PY

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "text_motion/real"
STATE = {"state": "waiting_for_alignment", "jobs": {}}


def save():
    f = OUT / "review_status.writing"
    f.write_text(json.dumps(STATE, indent=2) + "\n")
    f.replace(OUT / "review_status.json")


async def main():
    assert not (OUT / "review_status.json").exists()
    save()
    deadline = time.time() + 900
    while True:
        status = json.loads((OUT / "status.json").read_text())
        if status["state"] == "finished_needs_evaluation":
            break
        if status["state"] in ["failed", "preparation_failed"] or time.time() > deadline:
            STATE["state"] = "alignment_not_complete_no_evaluation"
            save()
            return
        await asyncio.sleep(5)
    semaphore = asyncio.Semaphore(2)
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")

    async def review(case):
        async with semaphore:
            with (OUT / "logs" / (case + "_review.log")).open("x") as log:
                p = await asyncio.create_subprocess_exec(PY, str(ROOT / "review_condmdi_real.py"), case,
                                                          env=env, stdout=log, stderr=log)
                STATE["jobs"][case] = dict(state="running", pid=p.pid, started=time.time())
                save()
                rc = await p.wait()
                STATE["jobs"][case].update(state="finished" if rc == 0 else "failed", exit_code=rc, finished=time.time())
                save()

    cases = json.loads((OUT / "PROTOCOL.json").read_text())["cases"]
    await asyncio.gather(*(review(c) for c in cases))
    STATE["state"] = "finished_needs_visual_review" if all(v["exit_code"] == 0 for v in STATE["jobs"].values()) else "failed"
    save()
    if STATE["state"] == "finished_needs_visual_review":
        items = "".join(f'<h2>{c}</h2><video controls preload="metadata" width="100%" src="cases/{c}/evaluation/comparison.mp4"></video><p><a href="cases/{c}/evaluation/comparison.jpg">固定五时刻对照图</a> · <a href="cases/{c}/evaluation/metrics.json">诊断指标</a></p>' for c in cases)
        (OUT / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>真实遮挡与正常动作：CondMDI</title><body style="max-width:1600px;margin:auto"><h1>真实开发六人：文字辅助补全</h1><p>观测输入｜原始 O0｜相机约束 O2｜CondMDI无文字+对齐｜CondMDI文字提议｜CondMDI文字+对齐。下排为固定尺度的根局部骨架前/侧视诊断；不是4DAnyone新生成的RGB，也没有真实隐藏动作标签。黄色点为可信候选，橙色为未知。没有修改前端生成模型权重。</p>' + items + '</body>')


if __name__ == "__main__":
    asyncio.run(main())
