import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "text_motion"


def save(state):
    tmp = OUT / "smoke_status.writing"
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(OUT / "smoke_status.json")


async def main():
    assert json.loads((OUT / "download.json").read_text())["state"] == "downloaded_crc_verified_not_loaded"
    scripts = ["smoke_condmdi.py", "condmdi_runtime.py", "condmdi_mask.py"]
    (OUT / "smoke_code_freeze.json").write_text(json.dumps(
        {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in scripts}, indent=2) + "\n")
    uuid = "GPU-0db9aa59-0293-a3ea-06a0-395913880fbd"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, FD_EXPECTED_GPU_UUID=uuid,
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
               PYTHONUNBUFFERED="1")
    with (OUT / "smoke.log").open("x") as log:
        process = await asyncio.create_subprocess_exec(
            "/mnt/ojc/miniconda3/envs/condmdi/bin/python", str(ROOT / "smoke_condmdi.py"),
            str(OUT / "weights/model000750000.pt"), env=env, stdout=log, stderr=log)
        state = dict(state="running", pid=process.pid, gpu_uuid=uuid, started=time.time())
        save(state)
        rc = await process.wait()
    state.update(state="finished" if rc == 0 else "failed", exit_code=rc, finished=time.time())
    save(state)


if __name__ == "__main__":
    asyncio.run(main())
