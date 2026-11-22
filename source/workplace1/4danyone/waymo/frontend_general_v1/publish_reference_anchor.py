"""Publish complete G2/G3 pairs only, reusing the validated CPU media renderer."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time
from publish_dev6_pairs import render, save, B, PY

A = B / "reference_anchor"
SUBJECTS = json.loads((A / "PROTOCOL.json").read_text())["subjects"]


def gallery():
    sections = []
    for sample in SUBJECTS:
        if not (A / sample / "publication.json").exists():
            sections.append(f"<section><h2>{sample}</h2><p>等待完整G2/G3与解码验收，无临时对照视频。</p></section>")
            continue
        links = []
        for branch in ("G2", "G3"):
            videos = " ".join(
                f'<a href="../{sample}/{branch}/outputs/turbo/videos/{v:02d}.mp4">{v:02d}{"*" if v in (4,9,14,19) else ""}</a>'
                for v in range(24))
            links.append(f"<p>{branch}：{videos}</p>")
        sections.append(
            f'<section><h2>{sample}</h2><video controls preload="metadata" src="{sample}/comparison.mp4"></video>'
            f'<p><a href="{sample}/comparison.jpg">首中末对照</a> · <a href="../{sample}/G3/validation.json">技术验收</a></p>'
            + "".join(links) + "</section>")
    page = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>G2/G3参考约束对照</title>
<style>body{font:16px/1.6 system-ui;max-width:1200px;margin:auto;padding:24px}section{padding:16px;background:#eee;margin:20px 0}video{width:100%}</style>
<h1>固定六人：G2原生成 / G3四参考约束</h1>
<p>G3沿用G2全部动作、输入、相机、四参考及Turbo权重；每步将相应目标latent约束到匹配噪声程度的RCP参考。
这是工程实验，不是论文原生配方，也没有质量通过。参考本身可能错误。</p>
<p>对照展示视角00和12，均不是强制约束视角。带*的04/09/14/19是被约束的参考视角，
不能把这些视角与RCP一致作为质量提高。其余20视角需独立评价。不可见动作和真实新视角没有真值。</p>
<p><a href="REPORT.md">方案说明</a> · <a href="status.json">运行状态</a> · <a href="../index.html">G1/G2完整对照</a></p>"""
    tmp = A / "index.writing.html"
    tmp.write_text(page + "".join(sections) + "</html>")
    tmp.replace(A / "index.html")


async def main():
    state = {"state": "waiting_complete_pairs", "jobs": {}}
    tasks = {}
    sem = asyncio.Semaphore(2)
    async def publish(sample):
        async with sem:
            with (A / "logs" / f"{sample}_publish.log").open("x") as log:
                proc = await asyncio.create_subprocess_exec(PY, str(Path(__file__)), "--sample", sample,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2"),
                    stdout=log, stderr=log)
                state["jobs"][sample] = dict(state="running", pid=proc.pid)
                save(A / "publication_status.json", state)
                rc = await proc.wait()
                state["jobs"][sample].update(state="completed" if rc == 0 else "failed", exit_code=rc)
                save(A / "publication_status.json", state)
    while True:
        for sample in SUBJECTS:
            if (A / sample / "publication.json").exists():
                state["jobs"].setdefault(sample, dict(state="completed_existing"))
            elif sample not in tasks and all((B / sample / branch / "validation.json").exists() for branch in ("G2", "G3")):
                tasks[sample] = asyncio.create_task(publish(sample))
        gallery()
        if all((A / s / "publication.json").exists() for s in SUBJECTS):
            break
        controller = json.loads((A / "status.json").read_text())
        if controller["state"] in ("failed_preserved_results", "complete_needs_quality_review"):
            await asyncio.gather(*tasks.values())
            break
        state["updated_at"] = time.time()
        save(A / "publication_status.json", state)
        await asyncio.sleep(20)
    await asyncio.gather(*tasks.values())
    state.update(state="finished_available_pairs", unpublished=[s for s in SUBJECTS if not (A / s / "publication.json").exists()])
    save(A / "publication_status.json", state)
    gallery()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", choices=SUBJECTS)
    args = parser.parse_args()
    if args.sample:
        render(args.sample, branches=("G2", "G3"), output_root=A / args.sample,
               branch_titles={"G2": "G2 original target", "G3": "G3 RCP anchored"})
    else:
        asyncio.run(main())
