"""Resume validated pilot packaging under the host tmux process supervisor."""
import asyncio
import json
import os
from pathlib import Path

from scan_native import ROOT, sha, write


async def main():
    semaphore = asyncio.Semaphore(2)
    status = {}
    async def one(row):
        async with semaphore:
            package = Path(row["sample_root"]).parent
            ready = package / "READY.json"
            if ready.exists():
                receipt = json.loads(ready.read_text())
                assert all(sha(package / v["path"]) == v["sha256"] for v in receipt["videos"])
                status[row["review_id"]] = {"state": "verified_existing"}
                return
            runtime = ROOT.parent / "round2/moving61_runtime" if row["output_slots"] == 61 else ROOT.parent.parent / "repo"
            env = dict(os.environ, FDANYONE_SAMPLE_ROOT=row["sample_root"],
                       FDANYONE_RUNTIME=str(runtime), FDANYONE_FRAMES=str(row["output_slots"]),
                       OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
            with (ROOT / "logs" / (row["review_id"]+"_finish_host.log")).open("x") as log:
                child = await asyncio.create_subprocess_exec(
                    "/mnt/ojc/miniconda3/envs/4danyone/bin/python", str(ROOT / "scripts/finish_input.py"),
                    env=env, stdout=log, stderr=log)
                code = await child.wait()
            status[row["review_id"]] = {"state": "completed" if code == 0 else "failed", "exit_code": code}
            write(ROOT / "pilot_packaging_status.json", status)
    await asyncio.gather(*(one(r) for r in json.loads((ROOT / "pilot_manifest.json").read_text())["candidates"]))
    write(ROOT / "pilot_packaging_status.json", status)
    assert all(v["state"] != "failed" for v in status.values())


if __name__ == "__main__":
    asyncio.run(main())
