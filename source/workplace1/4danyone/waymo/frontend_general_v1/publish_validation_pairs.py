"""Publish complete, verified G0/G2 pairs atomically; no quality claims."""
import argparse
import asyncio
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys
import time

import av
from PIL import Image, ImageDraw, ImageFont, ImageOps

G = Path(__file__).resolve().parent
V = G / "validation12"
B = V / "generation"
C = V / "comparison"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def atomic_json(path, value):
    tmp = path.with_suffix(".pending.json")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def ready(sample):
    return (B / "old_checks" / f"{sample}.json").exists() and (
        B / sample / "G2/validation.json").exists()


def render(sample):
    cfg = read(B / "PROTOCOL.json")
    assert sample in cfg["subjects"]
    old = read(B / "old_checks" / f"{sample}.json")
    new_root = B / sample / "G2"
    new = read(new_root / "validation.json")
    assert old["state"] == "old_outputs_decoded_reusable_as_whole_pipeline_baseline"
    assert new["state"] == "complete_decoding_and_stage_evidence_verified"
    target = [x for x in new["videos"] if x["stage"] == "target"]
    assert sorted(x["view"] for x in target) == list(range(24))
    count = cfg["frames"][sample]
    assert all(x["frames"] == count for x in target + old["videos"])
    source = Path(cfg["sources"][sample])
    paths = [
        source / "input/model_input.mp4",
        source / "videos/00.mp4", new_root / "outputs/turbo/videos/00.mp4",
        source / "videos/12.mp4", new_root / "outputs/turbo/videos/12.mp4",
    ]
    hashes = {str(p): digest(p) for p in paths}
    for view, path in ((0, paths[1]), (12, paths[3])):
        record = next(x for x in old["videos"] if x["view"] == view)
        assert record["sha256"] == hashes[str(path)], "Baseline changed"
    final = C / sample
    if final.exists():
        assert read(final / "manifest.json")["input_sha256"] == hashes
        return
    tmp = C / f".{sample}.building"
    tmp.mkdir(exist_ok=False)
    containers = []
    writer = None
    try:
        containers = [av.open(str(p)) for p in paths]
        for container in containers:
            container.streams.video[0].thread_count = 1
            assert container.streams.video[0].average_rate == 25
        decoders = [c.decode(video=0) for c in containers]
        font = ImageFont.truetype(
            str(Path(sys.prefix) / "lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"), 15)
        labels = ["Source model input", "G0 original / v00", "G2 candidate / v00",
                  "G0 original / v12", "G2 candidate / v12"]
        width, height, header = 264, 480, 52
        writer = av.open(str(tmp / "comparison.mp4"), mode="w")
        stream = writer.add_stream("libx264", rate=25)
        stream.width, stream.height = width * 5, height + header
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "preset": "fast", "threads": "2"}
        contact = []
        slots = sorted(set([0, count // 4, count // 2, 3 * count // 4, count - 1]))
        for frame_id in range(count):
            row = Image.new("RGB", (width * 5, height + header), (34, 34, 34))
            draw = ImageDraw.Draw(row)
            for column, decoder in enumerate(decoders):
                frame = next(decoder)
                assert frame.pts is not None
                assert abs(float(frame.pts * frame.time_base) - frame_id / 25) < 1e-3
                image = ImageOps.contain(frame.to_image(), (width, height))
                x = column * width + (width - image.width) // 2
                row.paste(image, (x, header + (height - image.height) // 2))
                draw.text((column * width + 5, 4), labels[column], font=font, fill="white")
            draw.text((5, 29), f"{sample}  t={frame_id / 25:.2f}s | Same time; cameras/crops may differ. Quality unapproved.",
                      font=font, fill=(255, 210, 110))
            if frame_id in slots:
                contact.append(row.copy())
            for packet in stream.encode(av.VideoFrame.from_image(row)):
                writer.mux(packet)
        for decoder in decoders:
            assert next(decoder, None) is None, "Unexpected extra frames"
        for packet in stream.encode():
            writer.mux(packet)
        writer.close()
        writer = None
        sheet = Image.new("RGB", (width * 5, (height + header) * len(contact)))
        for i, image in enumerate(contact):
            sheet.paste(image, (0, i * (height + header)))
        sheet.save(tmp / "contact.jpg", quality=90)
        with av.open(str(tmp / "comparison.mp4")) as check:
            check.streams.video[0].thread_count = 1
            assert sum(1 for _ in check.decode(video=0)) == count
        metadata = {
            "sample": sample, "frames": count, "fps": 25, "views_shown": [0, 12],
            "source_panel": "Model input, including temporal interpolation; not native raw video.",
            "native_video": str(source / "input/observed_native.mp4"),
            "input_sha256": hashes, "fixed_contact_frames": slots,
            "quality_approved": False, "manual_visual_review": "pending",
            "comparison": "Whole pipeline, same source timeline; cameras/crops may differ.",
            "validation_sha256": {
                str(B / "old_checks" / f"{sample}.json"): digest(B / "old_checks" / f"{sample}.json"),
                str(new_root / "validation.json"): digest(new_root / "validation.json"),
            },
            "output_sha256": {name: digest(tmp / name) for name in ("comparison.mp4", "contact.jpg")},
        }
        (tmp / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        tmp.rename(final)
    except BaseException:
        # Only this publisher's unpublished work is removed, never old results.
        if writer is not None:
            writer.close()
        shutil.rmtree(tmp)
        raise
    finally:
        for container in containers:
            container.close()


def index(subjects):
    entries = []
    for sample in subjects:
        if (C / sample / "manifest.json").exists():
            entries.append(f'<section><h2>{html.escape(sample)}</h2>'
                f'<video controls preload="none" width="100%" src="{sample}/comparison.mp4"></video>'
                f'<p><a href="{sample}/contact.jpg">固定五时刻对比图</a> · '
                f'<a href="{sample}/manifest.json">输入与校验记录</a></p></section>')
    if not entries:
        return
    page = """<!doctype html><meta charset="utf-8"><title>Val12 完整流程对照</title>
<style>body{max-width:1400px;margin:24px auto;background:#171717;color:#eee;font:17px sans-serif}
a{color:#9bd4ff}section{margin:36px 0}video{background:#222}</style>
<h1>Val12：原流程 G0 / 冻结候选 G2</h1>
<p>只发布旧、新各24视角均已完整解码的对象。每个视频展示输入和两个相反视角，
所有时间点均保留；五时刻拼图不代表人工逐帧验收。当前候选尚未通过质量验收。</p>
<p>第一列为实际模型输入（含时间插值），不是原生采集视频。左右方案相机与裁剪可能不同，
这是完整流程比较，不是相同相机下的像素精度比较。没有按效果挑选对象或时间窗口。</p>
"""
    tmp = C / ".index.pending.html"
    tmp.write_text(page + "\n".join(entries))
    tmp.replace(C / "index.html")


async def watch():
    C.mkdir(exist_ok=True)
    cfg = read(B / "PROTOCOL.json")
    state = {"state": "waiting_complete_pairs", "jobs": {}, "quality_approved": False}
    status_path = C / "status.json"
    semaphore = asyncio.Semaphore(2)
    tasks = {}

    async def one(sample):
        async with semaphore:
            log_path = C / f".{sample}.publish.log"
            with log_path.open("wb") as log:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, __file__, "--sample", sample, stdout=log, stderr=log,
                    env={**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"})
                state["jobs"][sample] = {"state": "running", "pid": proc.pid}
                atomic_json(status_path, state)
                code = await proc.wait()
            state["jobs"][sample] = {"state": "completed" if code == 0 else "failed",
                                     "exit_code": code}
            if code == 0:
                log_path.unlink()
            atomic_json(status_path, state)
            index(cfg["subjects"])

    while True:
        for sample in cfg["subjects"]:
            if sample not in tasks and ready(sample):
                tasks[sample] = asyncio.create_task(one(sample))
        generation = read(B / "status.json")
        terminal = generation["state"] != "generation_and_preparation_running"
        if terminal or len(tasks) == len(cfg["subjects"]):
            break
        state["updated_at"] = time.time()
        state["generation_state"] = generation["state"]
        atomic_json(status_path, state)
        await asyncio.sleep(30)
    await asyncio.gather(*tasks.values())
    state["unpublished"] = [s for s in cfg["subjects"] if not (C / s / "manifest.json").exists()]
    state["state"] = "complete_pairs_published_quality_pending" if not state["unpublished"] else "finished_with_unpublished"
    atomic_json(status_path, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample")
    args = parser.parse_args()
    if args.sample:
        render(args.sample)
    else:
        asyncio.run(watch())
