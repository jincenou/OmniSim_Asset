"""Explicit recovery of frame-count serialization failure, no repeated RIFE."""
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


async def main():
    protocol = json.loads((B / "MOTION_PROTOCOL.json").read_text())
    for name, digest in protocol["code_and_input_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    recovery = json.loads((B / "RECOVERY.json").read_text())
    for name, digest in recovery["code_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    state = json.loads((B / "motion_status.json").read_text())
    assert state["state"] == "failed" and state["jobs"]["interpolate"]["exit_code"] == 0
    source = B / "P18/calibrated/input"
    (source / "motion_pair").rename(source / "motion_pair_serialization_failed")
    gpu, uuid, lease = await gpu_lease(protocol["frames"], protocol)
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu),
               FDANYONE_SAMPLE_ROOT=str(source), FDANYONE_MOVING_ROOT=str(source),
               FDANYONE_RUNTIME=str(G.parents[1] / "repo"), FDANYONE_FRAMES=str(protocol["frames"]),
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("PYTHONPATH", None)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    state["first_attempt"] = dict(state["jobs"])
    state.update(state="recovering_motion", gpu=gpu, uuid=uuid, reason=recovery["reason"])
    def save():
        tmp = B / "motion_status.writing"
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(B / "motion_status.json")
    try:
        with (B / "logs/P18_motion_recovery.log").open("x") as log:
            started = time.time()
            p = await asyncio.create_subprocess_exec(
                "/mnt/ojc/miniconda3/envs/4danyone/bin/python", str(G / "context_motion.py"),
                env=env, cwd=G.parents[1] / "repo", stdout=log, stderr=log)
            state["pid"] = p.pid
            save()
            rc = await p.wait()
        state.update(state="context_motion_inferred_review_pending" if rc == 0 else "recovery_failed")
        state["recovery"] = dict(exit_code=rc, seconds=time.time()-started, pid=p.pid)
        state["scope"] = "Raw background retained, no SAM composite; legacy source motion report wording is not applicable. No repair or24-view generation yet."
        save()
    finally:
        fcntl.flock(lease, fcntl.LOCK_UN)
        lease.close()


if __name__ == "__main__":
    asyncio.run(main())
