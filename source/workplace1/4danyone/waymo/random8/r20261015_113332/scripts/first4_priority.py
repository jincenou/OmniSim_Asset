"""Prepare and reconstruct N01-N03 while the original queue completes N04.

The shared stage receipts prevent the original queue from launching duplicate
work. An immutable copy of every actual child exit receipt is kept separately
because the historical queue can subsequently reconcile an adopted process.
Only this controller's state file is written here; STATUS.json remains owned
by the original controller.
"""
import asyncio
from contextlib import asynccontextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sys

import controller_v3 as queue

R = Path(__file__).resolve().parents[1]
GPU_IDS = (1, 6, 7)
LABELS = ("N01", "N02", "N03")
STATE_PATH = R / "FIRST4_STATUS.json"
RECEIPTS = R / "logs" / "first4_receipts"
RECEIPTS.mkdir(exist_ok=True)
BASE_WRITE = queue.write
BASE_STAGE = queue.stage
SCRIPT_MAP = {
    "full_masks.py": "full_masks_priority.py",
    "train_ftgs.py": "train_ftgs_priority.py",
    "train_4dgs.py": "train_4dgs_priority.py",
    "evaluate_branch.py": "evaluate_branch_priority.py",
}


def write(path, value):
    BASE_WRITE(path, value)
    if path.parent == R / "logs" and value.get("state") in ("complete", "failed") and "exit_code" in value:
        target = RECEIPTS / f"{path.stem}_attempt{value.get('attempt', 0)}.json"
        if target.exists():
            assert json.loads(target.read_text()) == value, target
        else:
            BASE_WRITE(target, value)


def persist():
    queue.state.update(updated_at=queue.now(), disk_free_bytes=shutil.disk_usage(R).free)
    BASE_WRITE(STATE_PATH, queue.state)


@asynccontextmanager
async def lease(ids):
    assert ids and set(ids) <= set(GPU_IDS)
    handles = [(queue.OLD / "controller.lock").open("a"),
               (queue.W / "batch100/controller.lock").open("a")]
    handles.extend((queue.OLD / "gpu_locks" / f"gpu{gpu}.lock").open("a") for gpu in sorted(ids))
    try:
        while True:
            acquired = []
            try:
                for index, handle in enumerate(handles):
                    fcntl.flock(handle, (fcntl.LOCK_SH if index < 2 else fcntl.LOCK_EX) | fcntl.LOCK_NB)
                    acquired.append(handle)
            except BlockingIOError:
                for handle in acquired:
                    fcntl.flock(handle, fcntl.LOCK_UN)
                await asyncio.sleep(5)
            else:
                break
        yield tuple(handle.fileno() for handle in handles)
    finally:
        for handle in handles:
            handle.close()


async def choose_gpu(minimum):
    while True:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--id=" + ",".join(map(str, GPU_IDS)),
            "--query-gpu=index,memory.free,memory.used", "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE)
        output, _ = await proc.communicate()
        assert proc.returncode == 0
        devices = [tuple(map(int, line.split(b","))) for line in output.splitlines()]
        for gpu, free, used in sorted(devices, key=lambda item: (-item[1], item[0])):
            # New launches use cards that are actually idle, including context
            # memory; never assume a low GPU-utilization percentage means idle.
            if free < minimum or used > 100:
                continue
            with (queue.OLD / "gpu_locks" / f"gpu{gpu}.lock").open("a") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
            return gpu
        await asyncio.sleep(10)


async def stage(label, name, kind, script, *args, **kwargs):
    assert label in LABELS
    return await BASE_STAGE(label, name, kind, SCRIPT_MAP.get(script, script), *args, **kwargs)


async def guard():
    paused = {}
    while True:
        free = shutil.disk_usage(R).free
        jobs = {key: value for key, value in queue.state["jobs"].items()
                if key.split("/", 1)[0] in LABELS and value.get("state") == "running" and value.get("physical_gpus")}
        if free < int(35.75 * 2**30):
            for key, job in jobs.items():
                pid = job.get("pid")
                if pid and queue.alive(pid) and os.getpgid(pid) == pid and pid not in paused:
                    os.killpg(pid, signal.SIGSTOP)
                    paused[pid] = {"job": key, "pid": pid, "paused_at": queue.now()}
        if paused and free >= int(36.5 * 2**30):
            for pid in list(paused):
                if queue.alive(pid):
                    os.killpg(pid, signal.SIGCONT)
                del paused[pid]
        BASE_WRITE(R / "FIRST4_DISK_GUARD.json", {
            "state": "own_priority_jobs_paused_for_disk" if paused else "reserve_guard_active",
            "free_bytes": free, "paused_process_groups": list(paused.values()),
            "updated_at": queue.now()})
        persist()
        await asyncio.sleep(2)


async def main():
    lock = (R / "first4_priority.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert shutil.disk_usage(R).free >= 52 * 2**30, "Insufficient headroom for three shared datasets and frozen branches"
    queue.state = {
        "state": "first_four_priority_active", "controller_pid": os.getpid(),
        "priority_tracks": ["N01", "N02", "N03", "N04"],
        "owned_tracks": list(LABELS), "allowed_gpu_ids": list(GPU_IDS),
        "external_N04_controller_pid": json.loads((R / "STATUS.json").read_text())["controller_pid"],
        "jobs": {}, "samples": {}, "production_approved": False,
        "actual_exit_receipt_directory": str(RECEIPTS),
    }
    queue.write = write
    queue.persist = persist
    queue.lease = lease
    queue.choose_gpu = choose_gpu
    queue.stage = stage
    queue.MATERIALIZE = asyncio.Semaphore(3)
    queue.BACK_SEM = asyncio.Semaphore(3)
    cases = [case for case in queue.selection["main"] if case["sample_id"] in LABELS]
    assert len(cases) == 3
    for case in cases:
        label = case["sample_id"]
        proof = R / "samples" / label / "generation/research/generated_validation.json"
        assert json.loads(proof.read_text())["state"] == "all24_videos_decoded_same_retained_cameras"
        queue.state["samples"][label] = {"state": "shared_video_verified_reconstruction_pending"}
        for branch in ("ftgs_base5k", "ftgs_opt15k", "4dgs_grid25", "4dgs_grid61"):
            queue.state["jobs"][f"{label}/{branch}"] = {"state": "waiting_shared_RGBA_and_initialization"}
    persist()
    monitor = asyncio.create_task(guard())
    try:
        await asyncio.gather(*map(queue.reconstruct, cases))
        queue.state.update(state="first_four_supplement_registered_tasks_terminal_review_pending", ended_at=queue.now())
        persist()
    finally:
        monitor.cancel()
        await asyncio.gather(monitor, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
