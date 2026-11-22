"""Background source-only context inference with existing GPU leases."""
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from run_reference_anchor import gpu_lease

G = Path(__file__).resolve().parent
B = G / "text_motion/bilateral_context"
PY = "/mnt/ojc/miniconda3/envs/4danyone/bin/python"
REPO = G.parents[1] / "repo"
SCRIPTS = G.parent / "scene20/r20261019/scripts"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(data):
    tmp = B / "motion_status.writing"
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(B / "motion_status.json")


async def main():
    protocol_path = B / "MOTION_PROTOCOL.json"
    assert sha(protocol_path) == protocol_path.with_suffix(".sha256").read_text().strip()
    protocol = json.loads(protocol_path.read_text())
    for path, digest in protocol["code_and_input_sha256"].items():
        assert sha(path) == digest, path
    case = protocol["case"]
    source = B / case / "calibrated/input"
    if (B / "motion_status.json").exists():
        raise ValueError("Existing run; inspect before explicit recovery")
    state = {"state": "waiting_gpu", "sample": case, "quality_approved": False}
    save(state)
    gpu, uuid, lease = await gpu_lease(protocol["frames"], protocol)
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu),
               FDANYONE_SAMPLE_ID=case, FDANYONE_SAMPLE_ROOT=str(source),
               FDANYONE_MOVING_ROOT=str(source), FDANYONE_RUNTIME=str(REPO),
               FDANYONE_FRAMES=str(protocol["frames"]),
               FD_CONTEXT_NATIVE_FRAMES=str(protocol["native_frames"]),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    env.pop("PYTHONPATH", None)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    stages = [
        ("interpolate", B / "runtime/interpolate.py"),
        ("motion", SCRIPTS / "motion.py"),
    ]
    state.update(gpu=gpu, uuid=uuid, jobs={})
    (B / "logs").mkdir(exist_ok=True)
    try:
        for stage, script in stages:
            with (B / "logs" / f"{case}_{stage}.log").open("x") as log:
                started = time.time()
                process = await asyncio.create_subprocess_exec(
                    PY, str(script), cwd=REPO, env=env,
                    stdout=log, stderr=log)
                state.update(state="running", stage=stage, pid=process.pid)
                save(state)
                rc = await process.wait()
                state["jobs"][stage] = dict(exit_code=rc, seconds=time.time()-started, pid=process.pid)
                if rc:
                    state.update(state="failed")
                    save(state)
                    return
        state.update(state="context_motion_inferred_review_pending")
        state["scope"] = (
            "101 slots from 41 native frames; raw scene includes occluders. "
            "Existing generic motion report has a legacy SAM wording; no SAM "
            "composite was used here. No new 24-view generation or repaired motion yet.")
        save(state)
    finally:
        fcntl.flock(lease, fcntl.LOCK_UN)
        lease.close()


if __name__ == "__main__":
    asyncio.run(main())
