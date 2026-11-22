"""Bounded replay of the two original context prompts with safe CPU transfers."""
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
ROOT = B / "transfer_check"
STATE = {"state": "waiting_or_running", "jobs": {}, "quality_approved": False}


def save():
    STATE["updated_at"] = time.time()
    path = ROOT / "status.writing"
    path.write_text(json.dumps(STATE, indent=2) + "\n")
    path.replace(ROOT / "status.json")


async def job(arm, cfg):
    STATE["jobs"][arm] = {"state": "waiting_gpu"}
    save()
    gpu, uuid, lease = await gpu_lease(cfg["frames"], cfg)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    try:
        with (ROOT / f"{arm}.log").open("x") as log:
            process = await asyncio.create_subprocess_exec(
                "/mnt/ojc/miniconda3/envs/4danyone-sam2/bin/python",
                str(G / "segment_context_ready.py"), arm, env=env, stdout=log, stderr=log)
            STATE["jobs"][arm] = {"state": "running", "pid": process.pid,
                                  "gpu": gpu, "uuid": uuid, "started": time.time()}
            save()
            code = await process.wait()
        STATE["jobs"][arm].update(state="masks_ready_review_pending" if code == 0 else "failed", exit_code=code)
        save()
        assert code == 0, arm
    finally:
        fcntl.flock(lease, fcntl.LOCK_UN)
        lease.close()


async def main():
    assert not (ROOT / "status.json").exists()
    ROOT.mkdir(exist_ok=True)
    path = B / "TRANSFER_PROTOCOL.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == path.with_suffix(".sha256").read_text().strip()
    cfg = json.loads(path.read_text())
    for name, expected in cfg["code_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    save()
    results = await asyncio.gather(*(job(arm, cfg) for arm in ("box", "points")), return_exceptions=True)
    STATE["errors"] = [repr(r) for r in results if isinstance(r, BaseException)]
    STATE["state"] = "failed_preserved" if STATE["errors"] else "masks_ready_review_pending"
    save()


if __name__ == "__main__":
    asyncio.run(main())
