"""Bounded asynchronous root-condition probe, safe to run in independent tmux."""
import asyncio
import fcntl
import json
import os
from pathlib import Path
import time
from run_reference_anchor import gpu_lease

G = Path(__file__).resolve().parent
E = G / "text_motion/bilateral_context/root_condition_v1"
PY = "/mnt/ojc/miniconda3/envs/condmdi/bin/python"
STATE = dict(state="preparing", jobs={}, quality_approved=False)


def save():
    p = E / "status.writing"
    p.write_text(json.dumps(STATE, indent=2) + "\n")
    p.replace(E / "status.json")


async def run(key, stage, env, branch=None):
    command = [PY, str(G / "root_condition_probe.py"), stage]
    if branch:
        command += ["--branch", branch]
    with (E / "logs" / (key + ".log")).open("x") as log:
        start = time.time()
        child = await asyncio.create_subprocess_exec(*command, env=env, stdout=log, stderr=log)
        STATE["jobs"][key] = dict(state="running", pid=child.pid, started=start,
                                 uuid=env.get("FD_EXPECTED_GPU_UUID"))
        save()
        rc = await child.wait()
        STATE["jobs"][key].update(state="completed" if rc == 0 else "failed",
                                  exit_code=rc, seconds=time.time() - start)
        save()
        if rc:
            raise RuntimeError(f"{key} exited {rc}; preserved log")


async def main():
    assert not (E / "status.json").exists()
    (E / "logs").mkdir()
    cfg = json.loads((E / "PROTOCOL.json").read_text())
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               MKL_NUM_THREADS="2", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    try:
        await run("prepare", "prepare", dict(env, CUDA_VISIBLE_DEVICES=""))
        async def branch(name):
            gpu, uuid, lease = await gpu_lease(76, cfg)
            e = dict(env, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                     FD_EXPECTED_GPU_UUID=uuid)
            try:
                await run(name + "_sample", "sample", e, name)
                await run(name + "_retarget", "retarget", e, name)
                await run(name + "_align", "align", e, name)
            finally:
                fcntl.flock(lease, fcntl.LOCK_UN)
                lease.close()
        STATE["state"] = "sampling_and_aligning"
        save()
        results = await asyncio.gather(branch("no_text"), branch("text"), return_exceptions=True)
        failures = [str(r) for r in results if isinstance(r, BaseException)]
        STATE.update(state="failed" if failures else "completed_needs_motion_review", errors=failures)
    except Exception as exc:
        STATE.update(state="failed", errors=[repr(exc)])
    save()


if __name__ == "__main__":
    asyncio.run(main())
