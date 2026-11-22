"""Two concurrent motion priors, each with paired fixed alignment arms."""
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from run_reference_anchor import gpu_lease

G = Path(__file__).resolve().parent
E = G / "text_motion/bilateral_context/repair_v1"
PY = "/mnt/ojc/miniconda3/envs/condmdi/bin/python"
STATE = {"state": "preparing", "jobs": {}, "quality_approved": False}


def save():
    path = E / "status.writing"
    path.write_text(json.dumps(STATE, indent=2) + "\n")
    path.replace(E / "status.json")


async def run(key, arguments, env):
    with (E / "logs" / (key + ".log")).open("x") as log:
        start = time.time()
        process = await asyncio.create_subprocess_exec(PY, *map(str, arguments),
            env=env, stdout=log, stderr=log)
        STATE["jobs"][key] = dict(state="running", pid=process.pid, started=start,
                                  uuid=env.get("FD_EXPECTED_GPU_UUID"))
        save()
        rc = await process.wait()
        STATE["jobs"][key].update(state="completed" if rc == 0 else "failed",
                                  exit_code=rc, seconds=time.time()-start)
        save()
        return rc


async def main():
    assert not (E / "status.json").exists()
    protocol = json.loads((E / "PROTOCOL.json").read_text())
    assert hashlib.sha256((E / "PROTOCOL.json").read_bytes()).hexdigest() == (E / "PROTOCOL.sha256").read_text().strip()
    (E / "logs").mkdir()
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
               PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("PYTHONPATH", None)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    if await run("prepare", [G / "bilateral_condmdi.py", "prepare"], dict(env, CUDA_VISIBLE_DEVICES="")):
        STATE["state"] = "preparation_failed"
        save()
        return

    async def branch(name):
        gpu, uuid, lease = await gpu_lease(76, protocol)
        e = dict(env, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID", FD_EXPECTED_GPU_UUID=uuid)
        try:
            if await run(name+"_sample", [G / "bilateral_condmdi.py", "sample", "--branch", name], e):
                return
            if await run(name+"_retarget", [G / "bilateral_condmdi.py", "retarget", "--branch", name], e):
                return
            for arm in ("old_anchor", "gap_free"):
                if await run(name+"_"+arm, [G / "align_bilateral_motion.py", name, arm], e):
                    return
        finally:
            fcntl.flock(lease, fcntl.LOCK_UN)
            lease.close()

    STATE["state"] = "sampling_and_aligning"
    save()
    await asyncio.gather(*(branch(name) for name in ("no_text", "text")))
    STATE["state"] = "completed_needs_motion_review" if all(j["exit_code"] == 0 for j in STATE["jobs"].values()) else "failed"
    save()


if __name__ == "__main__":
    asyncio.run(main())
