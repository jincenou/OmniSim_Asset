"""Audit and crop the fixed native-source engineering pilot."""
import asyncio
import json
import os
from pathlib import Path
import time

from scan_native import write

ROOT = Path(__file__).resolve().parent
PY = "/mnt/ojc/miniconda3/envs/4danyone/bin/python"
RAW_PY = "/mnt/ojc/miniconda3/envs/4dgaussians/bin/python"


async def main():
    rows = json.loads((ROOT / "pilot_manifest.json").read_text())["candidates"]
    state = {"started": time.time(), "jobs": {}, "generation_started": False}
    path = ROOT / "pilot_prepare_status.json"
    if path.exists():
        raise RuntimeError("Explicit recovery required; will not overwrite an earlier attempt")
    semaphore = asyncio.Semaphore(2)

    async def one(row):
        async with semaphore:
            key = row["review_id"]
            env = dict(os.environ, FDANYONE_SAMPLE_ID=key, FDANYONE_SAMPLE_ROOT=row["sample_root"],
                       OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
            stages = [
                ("camera", [RAW_PY, str(ROOT / "scripts/raw_camera_audit.py"),
                            "--review-id", key, "--manifest", str(ROOT / "pilot_manifest.json")]),
                ("crop", [PY, str(ROOT / "scripts/m02_prepare.py")]),
            ]
            for stage, command in stages:
                with (ROOT / "logs" / f"{key}_{stage}.log").open("x") as log:
                    child = await asyncio.create_subprocess_exec(*command, env=env, stdout=log, stderr=log)
                    state["jobs"][key] = {"stage": stage, "state": "running", "pid": child.pid}
                    write(path, state)
                    code = await child.wait()
                state["jobs"][key].update(exit_code=code, state="completed" if code == 0 else "failed")
                write(path, state)
                if code:
                    return
    await asyncio.gather(*(one(row) for row in rows))
    state["finished"] = time.time()
    write(path, state)


if __name__ == "__main__":
    asyncio.run(main())
