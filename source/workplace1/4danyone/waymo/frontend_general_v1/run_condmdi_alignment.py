"""Bounded async retargeting + observation alignment, detached by tmux."""
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from run_condmdi_cases import GPU, PY

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "text_motion"
STATE = {"state": "starting", "jobs": {}}


def save():
    temp = OUT / "alignment_status.writing"
    temp.write_text(json.dumps(STATE, indent=2) + "\n")
    temp.replace(OUT / "alignment_status.json")


async def run(key, argv, env):
    with (OUT / "logs" / (key + ".log")).open("x") as log:
        p = await asyncio.create_subprocess_exec(PY, *map(str, argv), env=env, stdout=log, stderr=log)
        STATE["jobs"][key] = dict(state="running", pid=p.pid, started=time.time(),
                                  gpu_uuid=env["FD_EXPECTED_GPU_UUID"])
        save()
        rc = await p.wait()
        STATE["jobs"][key].update(state="finished" if rc == 0 else "failed",
                                  exit_code=rc, finished=time.time())
        save()
        return rc


async def main():
    assert not (OUT / "alignment_status.json").exists()
    frozen = json.loads((OUT / "retarget_code_freeze.json").read_text())
    assert all(hashlib.sha256((ROOT / n).read_bytes()).hexdigest() == h for n, h in frozen["files"].items())
    protocol = json.loads((OUT / "RETARGET_PROTOCOL.json").read_text())
    queue = asyncio.Queue()
    [queue.put_nowait((c, b)) for c in protocol["cases"] for b in ["no_text", "text"]]
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1")

    async def job(case, branch, uuid):
        e = dict(env, CUDA_VISIBLE_DEVICES="GPU-" + uuid, FD_EXPECTED_GPU_UUID="GPU-" + uuid)
        key = case + "_" + branch
        rc = await run(key + "_retarget", [ROOT / "retarget_condmdi.py", case, branch], e)
        if rc:
            return rc
        return await run(key + "_align", [
            ROOT / "optimize_motion.py", case.split("_")[0], "--device", "cuda",
            "--variant", "condmdi", "--condmdi-branch", branch,
            "--case-config", ROOT / "occlusion_v1" / case / "rgb/fit_config.json"], e)

    # First real case checks the exact model interface; no outcome-based tuning.
    c, b = queue.get_nowait()
    if await job(c, b, GPU[0]):
        STATE["state"] = "first_case_failed"
        save()
        return
    queue.task_done()
    STATE["state"] = "running"
    save()

    async def worker(uuid):
        while not queue.empty():
            c, b = queue.get_nowait()
            await job(c, b, uuid)
            queue.task_done()

    await asyncio.gather(*(worker(u) for u in GPU))
    STATE["state"] = "finished_needs_evaluation" if all(x["exit_code"] == 0 for x in STATE["jobs"].values()) else "failed"
    save()


if __name__ == "__main__":
    asyncio.run(main())
