import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path

from run_reference_anchor import gpu_lease

G = Path(__file__).resolve().parent
B = G / "text_motion/bilateral_context"


async def main():
    path = B / "MASK_POINTS_PROTOCOL.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == path.with_suffix(".sha256").read_text().strip()
    cfg = json.loads(path.read_text())
    for name, digest in cfg["code_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    gpu, uuid, lease = await gpu_lease(cfg["frames"], cfg)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    env.pop("PYTHONPATH", None)
    status = B / "mask_points_status.json"
    if status.exists():
        raise ValueError("Existing mask run")
    try:
        with (B / "logs/P18_context_masks_points.log").open("x") as log:
            p = await asyncio.create_subprocess_exec(
                "/mnt/ojc/miniconda3/envs/4danyone-sam2/bin/python", str(G / "segment_context_points.py"),
                env=env, stdout=log, stderr=log)
            state = dict(state="running", pid=p.pid, gpu=gpu, uuid=uuid, quality_approved=False)
            status.write_text(json.dumps(state, indent=2) + "\n")
            rc = await p.wait()
        state.update(state="masks_ready_review_pending" if rc == 0 else "failed", exit_code=rc)
        status.write_text(json.dumps(state, indent=2) + "\n")
    finally:
        fcntl.flock(lease, fcntl.LOCK_UN)
        lease.close()


if __name__ == "__main__":
    asyncio.run(main())
