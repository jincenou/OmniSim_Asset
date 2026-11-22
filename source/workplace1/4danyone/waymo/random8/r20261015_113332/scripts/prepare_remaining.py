"""Use an idle auxiliary card for shared inputs and numerical camera checks.

No optimizer is started here. Original queues retain sole responsibility for
all four frozen backend recipes. Prefix receipts and data are shared so that
those queues can adopt them instead of producing a second input dataset.
"""
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal

import controller_v3 as queue

R = Path(__file__).resolve().parents[1]
STATE = R / "AUX_STATUS.json"
EXISTING = ("N01", "N02", "N03", "N04")
PENDING = ("N06", "N07", "N08", "N11")
GPU = asyncio.Semaphore(1)
CPU = asyncio.Semaphore(2)
BASE_WRITE = queue.write


def write(path, value):
    BASE_WRITE(path, value)
    if path.parent == R / "logs" and value.get("pid") and value.get("exit_code") is not None:
        target = R / "logs/actual_receipts" / f"{path.stem}_pid{value['pid']}_exit{value['exit_code']}_aux.json"
        if not target.exists():
            BASE_WRITE(target, {"source": "AUX_STATUS.json", "job": path.stem, "actual_receipt": value})


def persist():
    queue.state.update(updated_at=queue.now(), disk_free_bytes=shutil.disk_usage(R).free)
    BASE_WRITE(STATE, queue.state)


async def guard():
    paused = {}
    while True:
        free = shutil.disk_usage(R).free
        if free < int(35.75 * 2**30):
            for key, job in queue.state["jobs"].items():
                pid = job.get("pid")
                if job.get("state") == "running" and job.get("physical_gpus") and pid and queue.alive(pid) and os.getpgid(pid) == pid and pid not in paused:
                    os.killpg(pid, signal.SIGSTOP)
                    paused[pid] = {"job": key, "pid": pid}
        if free >= int(36.5 * 2**30):
            for pid in list(paused):
                if queue.alive(pid):
                    os.killpg(pid, signal.SIGCONT)
                del paused[pid]
        BASE_WRITE(R / "AUX_DISK_GUARD.json", {"free_bytes": free, "paused_process_groups": list(paused.values()), "updated_at": queue.now()})
        await asyncio.sleep(2)


async def ready_input(case):
    label = case["sample_id"]
    base = R / "samples" / label
    try:
        assert json.loads((base / "generation/research/generated_validation.json").read_text())["state"] == "all24_videos_decoded_same_retained_cameras"
        if label in PENDING:
            proof = R / "research" / f"{label}-mask-complete.json"
            if not proof.exists():
                async with GPU, queue.disk_budget(label + "/aux_shared_RGBA", .4 if case["output_slots"] == 61 else .8):
                    queue.state["samples"][label]["state"] = "exporting_shared_RGBA_on_GPU5"
                    persist()
                    await queue.stage(label, "rgba", "sam", "full_masks.py", gpus=[5], minimum=10000, proof=proof)
            async with CPU:
                await queue.stage(label, "ftgs_initialization", "ftgs", "export_sequence.py", proof=base / "data/init.npz")
                await queue.stage(label, "4dgs_initialization", "ftgs", "export_4dgs.py", ["--label", label, "--data", base / "data"], proof=R / "four_dg/data" / label / "fdanyone.json")
        manifest = R / "four_dg/data" / label / "fdanyone.json"
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        proof = R / "research" / f"{label}_camera_projection.json"
        if not proof.exists():
            async with GPU, queue.disk_budget(label + "/aux_camera_projection", .01):
                queue.state["samples"][label]["state"] = "checking_actual_CUDA_camera_mapping"
                persist()
                await queue.stage(label, "camera_projection", "hust", "camera_projection_probe_v3.py", [label], gpus=[5], minimum=6500, proof=proof)
        result = json.loads(proof.read_text())
        assert result["state"] == "new_actual_data_all24_cameras_three_times_cuda_projection_passed"
        assert result["label"] == label and result["max_error_px"] < .001
        assert hashlib.sha256(manifest.read_bytes()).hexdigest() == digest
        queue.state["samples"][label].update(state="shared_inputs_camera_verified_no_new_optimizer_started", camera_proof=str(proof), input_manifest_sha256=digest)
    except Exception as error:
        logs = "\n".join(Path(value["log"]).read_text(errors="replace")[-5000:] for key, value in queue.state["jobs"].items() if key.startswith(label + "/") and value.get("state") == "failed" and value.get("log"))
        hard = "Empty foreground: no legal common RGBA" in str(error) + logs
        queue.state["samples"][label].update(state="common_RGBA_hard_failure_kept_in_denominator" if hard else "auxiliary_failure_requires_review_no_person_replacement", reason=f"{type(error).__name__}:{error}", reserve_eligible=hard)
    persist()


async def main():
    lock = (R / "auxiliary.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not STATE.exists(), "No silent restart; inspect existing auxiliary receipt first"
    queue.state = {"state": "shared_input_camera_preparation_active", "controller_pid": os.getpid(), "gpu_ids": [5],
                   "scope": "Existing frozen eight actors only; shared-input preprocessing and numerical diagnostics, never optimization.",
                   "jobs": {}, "samples": {label: {"state": "waiting_auxiliary_admission"} for label in EXISTING + PENDING}}
    queue.write = write
    queue.persist = persist
    persist()
    monitor = asyncio.create_task(guard())
    try:
        registered = {case["sample_id"]: case for case in queue.selection["main"] + queue.selection["reserve"]}
        await asyncio.gather(*map(lambda label: ready_input(registered[label]), EXISTING + PENDING))
        queue.state.update(state="shared_input_camera_preparation_terminal_review_pending", ended_at=queue.now())
        persist()
    finally:
        monitor.cancel()
        await asyncio.gather(monitor, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
