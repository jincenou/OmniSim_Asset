"""Publish only complete validated pairs; bounded CPU tasks, no interim comparison movies."""
import argparse
import asyncio
import html
import json
import os
from pathlib import Path
import time
import av
from PIL import Image, ImageDraw, ImageOps

G = Path(__file__).resolve().parent
B = G / "generation_dev6"
PY = "/mnt/ojc/miniconda3/envs/4danyone/bin/python"
SUBJECTS = json.loads((B / "PROTOCOL.json").read_text())["subjects"]


def save(path, data):
    tmp = path.with_suffix(".writing")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def render(sample, branches=("G1", "G2"), output_root=None, branch_titles=None):
    root = B / sample
    destination = root if output_root is None else Path(output_root)
    destination.mkdir(exist_ok=True, parents=True)
    assert len(branches) == 2
    titles = branch_titles or {"G1": "G1 original pose", "G2": "G2 completion"}
    for branch in branches:
        checked = json.loads((root / branch / "validation.json").read_text())
        assert checked["state"] == "complete_decoding_and_stage_evidence_verified"
    n = json.loads((root / "pairing.json").read_text())["frames"]
    paths = [root / branches[0] / "inputs/main.mp4"] + [
        root / branch / "outputs/turbo/videos" / f"{view:02d}.mp4"
        for view in (0, 12) for branch in branches]
    labels = ["Input video (pre-crop)"] + [
        f"{titles[branch]} v{view:02d}" for view in (0, 12) for branch in branches]
    final = destination / "comparison.mp4"
    assert not final.exists()
    tmp = destination / "comparison.writing.mp4"
    containers = [av.open(str(path)) for path in paths]
    iterators = [container.decode(video=0) for container in containers]
    selected = {}
    count = 0
    try:
        with av.open(str(tmp), "w") as output:
            stream = output.add_stream("libx264", rate=25)
            stream.width, stream.height = 1120, 432
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18", "threads": "2", "preset": "medium"}
            for t, frames in enumerate(zip(*iterators, strict=True)):
                panel = Image.new("RGB", (1120, 432), "#222222")
                draw = ImageDraw.Draw(panel)
                for col, (frame, label) in enumerate(zip(frames, labels, strict=True)):
                    im = ImageOps.contain(frame.to_image(), (224, 408))
                    panel.paste(im, (col * 224 + (224 - im.width) // 2, 24 + (408 - im.height) // 2))
                    draw.text((col * 224 + 3, 6), label, fill="white")
                if t in (0, n // 2, n - 1):
                    selected[t] = panel.copy()
                for packet in stream.encode(av.VideoFrame.from_image(panel)):
                    output.mux(packet)
                count += 1
            for packet in stream.encode():
                output.mux(packet)
        assert count == n, (count, n)
        with av.open(str(tmp)) as decoded:
            assert sum(1 for _ in decoded.decode(video=0)) == n
        sheet = Image.new("RGB", (1120, 432 * 3), "#222222")
        for row, t in enumerate((0, n // 2, n - 1)):
            sheet.paste(selected[t], (0, row * 432))
        sheet.save(destination / "comparison.jpg", quality=94)
        tmp.replace(final)
        save(destination / "publication.json", {
            "state": "complete_pair_published", "frames": n, "fps": 25,
            "columns": labels, "views": [0, 12], "contact_frames": [0, n // 2, n - 1],
            "branches": list(branches),
            "quality_approved": False, "manual_visual_review": "pending",
            "scope": "Actual model input versus two candidate generators, not hidden motion truth"})
    finally:
        for container in containers:
            container.close()


def gallery():
    sections = []
    for sample in SUBJECTS:
        root = B / sample
        if not (root / "publication.json").exists():
            sections.append(f"<section><h2>{sample}</h2><p>对照尚未完整发布；准备、生成或检查中的输出不算成品。</p></section>")
            continue
        links = []
        for label in ("G1", "G2"):
            videos = " ".join(f'<a href="{sample}/{label}/outputs/turbo/videos/{i:02d}.mp4">{i:02d}</a>' for i in range(24))
            evidence = json.loads((root / label / "stage_evidence/report.json").read_text())
            rcp = " ".join(f'<a href="{sample}/{label}/stage_evidence/rcp/{i:02d}.mp4">v{v:02d}</a>'
                           for i, v in enumerate(evidence["rcp_camera_ids"]))
            links.append(f"<p><b>{label}完整24路：</b>{videos}<br>真实RCP：{rcp}</p>")
        sections.append(
            f'<section><h2>{sample}</h2><video controls preload="metadata" src="{sample}/comparison.mp4"></video>'
            f'<p><a href="{sample}/comparison.mp4">下载对比视频</a> · '
            f'<a href="{sample}/comparison.jpg">首/中/末对比图</a> · '
            f'<a href="{sample}/pairing.json">配对检查</a></p>'
            + "".join(links) + "</section>")
    page = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>遮挡补全与24视角生成</title>
<style>body{font:16px/1.65 system-ui;max-width:1200px;margin:auto;padding:24px;background:#f5f5f5}section{background:white;padding:20px;margin:20px 0}video{width:100%;max-height:600px}a{color:#1266ae}</style>
<h1>固定六人：原动作条件 / 补全动作条件</h1>
<p><strong>P18 质量复核：遮挡动作补全未成功，仍有明显怪异姿势。此前“部分改善”的判断撤回。
这里保留失败对照；不因膝角变小或鞋更可见而判动作修复有效。</strong></p>
<p>仅发布两组完成全帧验证后的对照。G1：原动作、可靠固定规范参考；G2：CondMDI人工文字提议加相机约束后的动作。
相同输入、目标相机、seed42与Turbo权重。此处是完整条件流程比较，不能将全部差异归因于prompt。
仍有动作保护线失败及未知隐藏动作，不是质量通过或真实新视角精度。</p>
<p>视频列：传入模型的视频（内部source裁剪前） → G1/G2视角00 → G1/G2视角12。两个生成视角相对180°，不保证分别是真实正面/背面。
所有24路与RCP另有入口；预编码PNG保存在各组stage_evidence目录。
<a href="REPORT.md">过程说明</a> · <a href="status.json">原队列</a> ·
<a href="P04_geometry_repair/status.json">P04几何修复队列</a></p>"""
    temp = B / "index.writing.html"
    temp.write_text(page + "".join(sections) + "</html>")
    temp.replace(B / "index.html")


async def main():
    state = {"state": "waiting_for_complete_pairs", "jobs": {}}
    sem = asyncio.Semaphore(2)
    tasks = {}
    async def publish(sample):
        async with sem:
            with (B / "logs" / f"{sample}_publish.log").open("x") as log:
                proc = await asyncio.create_subprocess_exec(
                    PY, str(Path(__file__)), "--sample", sample, stdout=log, stderr=log,
                    env=dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2"))
                state["jobs"][sample] = dict(state="running", pid=proc.pid)
                save(B / "publication_status.json", state)
                rc = await proc.wait()
                state["jobs"][sample].update(state="completed" if rc == 0 else "failed", exit_code=rc)
                save(B / "publication_status.json", state)
                gallery()
    while True:
        for sample in SUBJECTS:
            root = B / sample
            if (root / "publication.json").exists():
                state["jobs"].setdefault(sample, {"state": "completed_existing"})
            elif sample not in tasks and all((root / label / "validation.json").exists() for label in ("G1", "G2")):
                tasks[sample] = asyncio.create_task(publish(sample))
        gallery()
        if all(sample in state["jobs"] and state["jobs"][sample]["state"].startswith("completed") for sample in SUBJECTS):
            break
        original = json.loads((B / "status.json").read_text())
        repair_path = B / "P04_geometry_repair/status.json"
        repair = json.loads(repair_path.read_text()) if repair_path.exists() else {"state": "not_started"}
        terminal = ("queue_failed_preserved_results", "generated_validated_needs_visual_review", "controller_failed", "repair_failed")
        if original["state"] in terminal and repair["state"] in terminal:
            await asyncio.gather(*tasks.values())
            break
        state["updated_at"] = time.time()
        save(B / "publication_status.json", state)
        await asyncio.sleep(20)
    await asyncio.gather(*tasks.values())
    state["state"] = "finished_available_pairs_needs_visual_review"
    state["unpublished"] = [sample for sample in SUBJECTS if not (B / sample / "publication.json").exists()]
    save(B / "publication_status.json", state)
    gallery()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", choices=SUBJECTS)
    args = parser.parse_args()
    if args.sample:
        render(args.sample)
    else:
        asyncio.run(main())
