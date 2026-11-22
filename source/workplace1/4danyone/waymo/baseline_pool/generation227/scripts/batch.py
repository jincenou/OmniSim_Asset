"""Bounded async inference with the existing per-GPU leases; source packages read-only."""
import asyncio
import collections
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT.parent
WAYMO = POOL.parent
PYTHON = Path("/mnt/ojc/miniconda3/envs/4danyone/bin/python")
LOCKS = WAYMO / "gpu_leases_20261019"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".writing")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def output_dir(row):
    c = row["candidate"]
    return ROOT / "scenes" / c["split"] / c["scene"] / c["review_id"]


def publish(state, rows):
    state["updated_at"] = time.time()
    state["counts"] = dict(collections.Counter(j["state"] for j in state["jobs"].values()))
    write(ROOT / "STATUS.json", state)
    blocks = [
        '<!doctype html><meta charset="utf-8"><title>Waymo 227条轨迹 · 4DAnyone</title>',
        '<style>body{max-width:1200px;margin:24px auto;background:#171717;color:#eee;font-family:sans-serif}a{color:#9df}video{max-width:95%;max-height:700px}img{max-width:100%}article{border-top:1px solid #777;padding:18px 0}</style>',
        "<h1>Waymo → 原基线 4DAnyone Turbo</h1>",
        "<p>固定227条候选轨迹；未知跨相机身份未强行合并，不等于227个已确认独立人物。"
        "公开权重推理，24视角、25fps、704×1280。仅完整解码通过的结果列入效果区；生成完成不代表视觉质量合格。</p>",
        f"<p>状态：{html.escape(str(state['counts']))}；未领取 {len(rows)-len(state['jobs'])}。"
        '<a href="../index.html">输入与骨架池</a> · <a href="STATUS.json">实时状态</a> · <a href="PLAN.md">执行说明</a></p>',
    ]
    for row in rows:
        job = state["jobs"].get(row["id"], {})
        if job.get("state") != "complete":
            continue
        base = output_dir(row)
        assert (base / "manifest.json").is_file()
        rel = str(base.relative_to(ROOT))
        source = Path(os.path.relpath(Path(row["package"]) / "clip0", ROOT))
        notes = row["review"].get("state", "pending")
        blocks.append(
            f'<article><h2>{html.escape(row["id"])}</h2><p>输入复核：{html.escape(notes)}；生成视觉复核待完成。</p>'
            f'<a href="{source}/observed_native.mp4">真实观测</a> · '
            f'<a href="{source}/inputs/main.mp4">实际输入</a> · '
            f'<a href="{source}/motion_pair/projection_comparison.mp4">骨架对照</a> · '
            f'<a href="{rel}/contact.jpg">首中末六视角图</a> · '
            f'<a href="{rel}/manifest.json">来源/校验</a>'
            f'<br><video controls preload="none" src="{rel}/preview.mp4"></video>'
            f'<details><summary>24个原始视角</summary>'
            + " · ".join(f'<a href="{rel}/videos/{v:02d}.mp4">{v:02d}</a>' for v in range(24))
            + "</details></article>"
        )
    failed = {k: j for k, j in state["jobs"].items() if j["state"] == "failed"}
    if failed:
        blocks.append("<h2>失败记录（未静默换人）</h2><pre>" + html.escape(
            json.dumps(failed, ensure_ascii=False, indent=2)) + "</pre>")
    temp = ROOT / "index.writing"
    temp.write_text("\n".join(blocks))
    temp.replace(ROOT / "index.html")


def stage_work(row):
    base = output_dir(row)
    base.mkdir(parents=True, exist_ok=False)
    source = Path(row["package"])
    assert sha(source / "READY.json") == row["ready_sha256"]
    gen = base / ".work/generation"
    (gen / "inputs").mkdir(parents=True)
    (gen / "preflight").mkdir()
    (gen / "research").mkdir()
    for name in ("main.mp4", "frame_mapping.json"):
        (gen / "inputs" / name).symlink_to(source / "clip0/inputs" / name)
    for name in ("motion", "conditioning", "root_motion.npz"):
        (gen / "preflight" / name).symlink_to(source / "conditions/preflight" / name)
    root_record = source / "conditions/research/root_local.json"
    if root_record.is_file():
        (gen / "research/root_local.json").symlink_to(root_record)
    else:
        evidence_path = source / "conditions/research/empty_mask_retention.json"
        evidence = json.loads(evidence_path.read_text())
        assert evidence["changed_model_outputs"] is False
        assert all(c["all_joints_all_frames_inside"] for c in evidence["coverage"])
        assert evidence["source_mask_retention_min_on_valid_frames"] >= .95
        write(gen / "research/root_local.json", {
            "state": "saved_root_local_conditions_reused_after_empty_mask_NA_audit",
            "record_origin": "Reconstructed provenance only; original preparation exited before writing root_local.json",
            "retention_evidence": str(evidence_path), "retention_evidence_sha256": sha(evidence_path),
            "source_video_sha256": sha(source / "clip0/inputs/main.mp4"),
            "root_motion_sha256": sha(source / "conditions/preflight/root_motion.npz"),
            "source_motion_reused_without_modification": True,
            "source_rigid_recomposition_error": None,
            "world_frame": "root_local_human_world", "quality_approved": False,
            "unknown_mask_frames": evidence["source_mask_unknown_frames"],
        })
    write(gen / "preflight/review.json", {
        "allow_generation": True,
        "authorization": "2026-10-20 user requested generation of prepared227 for visual inspection",
        "technical_ready_sha256": row["ready_sha256"],
        "input_review": row["review"], "quality_approved": False,
        "scope": "Experimental output inspection; partial occlusion is not an automatic exclusion",
    })
    write(base / "input_reference.json", row)
    return base, gen


async def main():
    singleton = (ROOT / "controller.lock").open("a")
    fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = json.loads((ROOT / "manifest.json").read_text())["rows"]
    status_path = ROOT / "STATUS.json"
    state = json.loads(status_path.read_text()) if status_path.exists() else {
        "state": "running", "started_at": time.time(), "target": len(rows),
        "training": False, "jobs": {}, "workers": {},
    }
    # Resume only never-started jobs. Interrupted work requires an explicit recovery.
    assert not any(j["state"] in ("generating", "validating") for j in state["jobs"].values()), \
        "Existing running state requires process/recovery audit"
    state["state"] = "running"
    frozen = {str(p): sha(p) for p in [ROOT / "manifest.json", *sorted((ROOT / "scripts").glob("*.py"))]}
    lock_file = ROOT / "code_frozen.json"
    if lock_file.exists():
        assert json.loads(lock_file.read_text()) == frozen
    else:
        write(lock_file, frozen)
    queue = asyncio.Queue()
    tuple(map(queue.put_nowait, (r for r in rows if r["id"] not in state["jobs"])))
    cpu = asyncio.Semaphore(2)

    async def run_child(script, args, env, row, gpu, stage):
        assert all(sha(Path(p)) == h for p, h in frozen.items()), "Frozen generation code changed"
        with (ROOT / "logs" / f"{row['id']}_{stage}.log").open("x") as log:
            child = await asyncio.create_subprocess_exec(str(PYTHON), str(ROOT / "scripts" / script),
                *args, env=env, stdout=log, stderr=log)
            state["jobs"][row["id"]].update(state=stage, gpu=gpu, pid=child.pid,
                                            stage_started=time.time())
            publish(state, rows)
            return await child.wait()

    async def worker(gpu):
        while not queue.empty():
            state["workers"][str(gpu)] = "waiting_for_shared_lease"
            publish(state, rows)
            lease = (LOCKS / f"gpu{gpu}.lock").open("a")
            row = None
            try:
                await asyncio.to_thread(fcntl.flock, lease, fcntl.LOCK_EX)
                process = await asyncio.create_subprocess_exec("nvidia-smi", f"--id={gpu}",
                    "--query-gpu=memory.free", "--format=csv,noheader,nounits",
                    stdout=asyncio.subprocess.PIPE)
                result, _ = await process.communicate()
                # Each active job reserves 1GiB transient disk headroom. Input preparation
                # independently retains its 8GiB guard; this is not the canceled35GiB floor.
                active = sum(j["state"] in ("generating", "validating") for j in state["jobs"].values())
                if process.returncode or int(result.strip()) < 35000 or shutil.disk_usage(ROOT).free < (4 + active + 1)*2**30:
                    state["workers"][str(gpu)] = "waiting_for_memory_or_disk"
                    publish(state, rows)
                    await asyncio.sleep(20)
                    continue
                try:
                    row = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                state["jobs"][row["id"]] = {"state": "admitted", "gpu": gpu, "started_at": time.time()}
                base, gen = stage_work(row)
                runtime = WAYMO / "round2/moving61_runtime" if row["candidate"]["output_slots"] == 61 else WAYMO.parent / "repo"
                env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu),
                    OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
                    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONNOUSERSITE="1",
                    PYTHONDONTWRITEBYTECODE="1", FDANYONE_WAYMO_ROOT=str(gen),
                    FDANYONE_RUNTIME=str(runtime), FDANYONE_RETAINED_VIEWS="24",
                    TORCH_EXTENSIONS_DIR=str(POOL / "cache/torch_extensions"),
                    TORCH_CUDA_ARCH_LIST="8.9", MAX_JOBS="2", TMPDIR=str(ROOT / "tmp"))
                env.pop("PYTHONPATH", None)
                env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
                code = await run_child("generate_single.py", ["turbo"], env, row, gpu, "generating")
                if code:
                    raise RuntimeError(f"Generation exited {code}; see log; no automatic profile change")
                state["jobs"][row["id"]]["generation_finished_at"] = time.time()
            except Exception as error:
                if row is None:
                    raise
                state["jobs"][row["id"]].update(state="failed", error=repr(error), finished_at=time.time())
                publish(state, rows)
                continue
            finally:
                fcntl.flock(lease, fcntl.LOCK_UN)
                lease.close()
            # The generation process has exited; CPU validation does not hold a GPU lease.
            try:
                async with cpu:
                    code = await run_child("deliver.py", [str(base), str(row["candidate"]["output_slots"])],
                                           env, row, gpu, "validating")
                    if code:
                        raise RuntimeError(f"Delivery exited {code}; generated output retained")
                state["jobs"][row["id"]].update(state="complete", finished_at=time.time())
            except Exception as error:
                state["jobs"][row["id"]].update(state="failed", error=repr(error), finished_at=time.time())
            publish(state, rows)
    publish(state, rows)
    await asyncio.gather(*(worker(gpu) for gpu in range(8)))
    state["state"] = "finished_with_failures" if any(j["state"] == "failed" for j in state["jobs"].values()) else "finished"
    state["finished_at"] = time.time()
    publish(state, rows)


if __name__ == "__main__":
    asyncio.run(main())
