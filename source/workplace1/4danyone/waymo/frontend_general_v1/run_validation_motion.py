"""Bounded async validation preparation and GPU motion inference under tmux."""
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from run_reference_anchor import gpu_lease

G = Path(__file__).resolve().parent
V = G / "validation12"
PY = "/mnt/ojc/miniconda3/envs/condmdi/bin/python"
STATE = dict(state="preparing", jobs={}, quality_approved=False)
CPU = asyncio.Semaphore(2)


def save():
    STATE["updated_at"] = time.time()
    p = V / "motion/status.writing"
    p.write_text(json.dumps(STATE, indent=2) + "\n")
    p.replace(V / "motion/status.json")


async def run(sample, stage, env):
    args = ([G / "align_validation_motion.py", sample] if stage == "align" else
            [G / "validation_motion.py", sample, stage])
    key = sample + "_" + stage
    with (V / "motion/logs" / (key + ".log")).open("x") as log:
        started = time.time()
        child = await asyncio.create_subprocess_exec(
            PY, *map(str, args), env=env, stdout=log, stderr=log)
        STATE["jobs"][key] = dict(state="running", pid=child.pid, started=started,
                                 uuid=env.get("FD_EXPECTED_GPU_UUID"))
        save()
        rc = await child.wait()
        STATE["jobs"][key].update(state="completed" if rc == 0 else "failed",
                                  exit_code=rc, seconds=time.time() - started)
        save()
        if rc:
            raise RuntimeError(f"{key} exited {rc}")


async def main():
    assert not (V / "motion/status.json").exists()
    (V / "motion/logs").mkdir(parents=True, exist_ok=False)
    path = V / "MOTION_PROTOCOL.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == path.with_suffix(".sha256").read_text().strip()
    cfg = json.loads(path.read_text())
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    async def subject(sample):
        async with CPU:
            await run(sample, "prepare", dict(env, CUDA_VISIBLE_DEVICES=""))
        STATE["jobs"][sample + "_queue"] = dict(state="waiting_gpu")
        save()
        gpu, uuid, lease = await gpu_lease(cfg["frames"][sample], cfg)
        e = dict(env, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                 FD_EXPECTED_GPU_UUID=uuid)
        STATE["jobs"][sample + "_queue"] = dict(state="gpu_acquired", gpu=gpu, uuid=uuid)
        save()
        try:
            await run(sample, "sample", e)
            await run(sample, "retarget", e)
            await run(sample, "align", e)
            STATE["jobs"][sample + "_queue"]["state"] = "completed_needs_review"
            save()
        finally:
            fcntl.flock(lease, fcntl.LOCK_UN)
            lease.close()

    STATE["state"] = "motion_validation_running"
    save()
    outcomes = await asyncio.gather(*(subject(s) for s in cfg["cases"]), return_exceptions=True)
    errors = {s: repr(result) for s, result in zip(cfg["cases"], outcomes) if isinstance(result, BaseException)}
    STATE.update(state="completed_with_failures" if errors else "completed_needs_motion_review", errors=errors)
    save()


if __name__ == "__main__":
    asyncio.run(main())
