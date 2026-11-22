"""Two bounded 3D-assisted alignment probes; do not change monocular controls."""
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
STATE = {"state": "waiting_or_running", "jobs": {}, "quality_approved": False}


def save():
    p = E / "box_status.writing"
    p.write_text(json.dumps(STATE, indent=2) + "\n")
    p.replace(E / "box_status.json")


async def main():
    path = E / "BOX_PROTOCOL.json"
    assert not (E / "box_status.json").exists()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == path.with_suffix(".sha256").read_text().strip()
    cfg = json.loads(path.read_text())
    for name, digest in cfg["code_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    async def run(branch):
        gpu, uuid, lease = await gpu_lease(76, cfg)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   FD_EXPECTED_GPU_UUID=uuid, OMP_NUM_THREADS="2",
                   OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
        env.pop("PYTHONPATH", None)
        started = time.time()
        try:
            with (E / "logs" / (branch+"_box_assisted.log")).open("x") as log:
                p = await asyncio.create_subprocess_exec(
                    "/mnt/ojc/miniconda3/envs/condmdi/bin/python", str(G / "align_bilateral_boxes.py"),
                    branch, "box_assisted", env=env, stdout=log, stderr=log)
                STATE["jobs"][branch] = dict(state="running", pid=p.pid, gpu=gpu, uuid=uuid)
                save()
                rc = await p.wait()
            STATE["jobs"][branch].update(state="complete" if rc == 0 else "failed",
                                        exit_code=rc, seconds=time.time()-started)
            save()
        finally:
            fcntl.flock(lease, fcntl.LOCK_UN)
            lease.close()
    await asyncio.gather(*(run(branch) for branch in ("no_text", "text")))
    STATE["state"] = "completed_needs_review" if all(v["exit_code"] == 0 for v in STATE["jobs"].values()) else "failed"
    save()


if __name__ == "__main__":
    asyncio.run(main())
