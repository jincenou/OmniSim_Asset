"""Apply the unchanged prompt proposal recipe to six real development clips."""
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from run_condmdi_cases import GPU, PY
from audit_condmdi_retarget import audit

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "text_motion/real"
STATE = {"state": "preparing", "jobs": {}}


def save():
    tmp = OUT / "status.writing"
    tmp.write_text(json.dumps(STATE, indent=2) + "\n")
    tmp.replace(OUT / "status.json")


async def run(key, argv, env):
    with (OUT / "logs" / (key + ".log")).open("x") as log:
        p = await asyncio.create_subprocess_exec(PY, *map(str, argv), env=env, stdout=log, stderr=log)
        STATE["jobs"][key] = dict(state="running", pid=p.pid, started=time.time(),
                                  gpu_uuid=env.get("FD_EXPECTED_GPU_UUID"))
        save()
        rc = await p.wait()
        STATE["jobs"][key].update(state="finished" if rc == 0 else "failed",
                                  exit_code=rc, finished=time.time())
        save()
        return rc


async def main():
    assert not (OUT / "status.json").exists()
    frozen = json.loads((OUT / "code_freeze.json").read_text())
    assert all(hashlib.sha256((ROOT / n).read_bytes()).hexdigest() == h for n, h in frozen["files"].items())
    protocol = json.loads((OUT / "PROTOCOL.json").read_text())
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    semaphore = asyncio.Semaphore(2)

    async def prepare(case):
        async with semaphore:
            rc = await run(case + "_prepare", [ROOT / "prepare_condmdi_case.py", case, "--suite", "real"], env)
            if rc == 0:
                audit(OUT / "cases" / case)
            return rc

    if any(await asyncio.gather(*(prepare(c) for c in protocol["cases"]))):
        STATE["state"] = "preparation_failed"
        save()
        return
    queue = asyncio.Queue()
    [queue.put_nowait((c, b)) for c in protocol["cases"] for b in ["no_text", "text"]]
    STATE["state"] = "running"
    save()

    async def worker(uuid):
        e = dict(env, CUDA_VISIBLE_DEVICES="GPU-" + uuid, FD_EXPECTED_GPU_UUID="GPU-" + uuid)
        while not queue.empty():
            case, branch = queue.get_nowait()
            key = case + "_" + branch
            rc = await run(key + "_sample", [ROOT / "sample_condmdi_case.py", case, branch, "--suite", "real"], e)
            if rc == 0:
                rc = await run(key + "_retarget", [ROOT / "retarget_condmdi.py", case, branch, "--suite", "real"], e)
            if rc == 0:
                await run(key + "_align", [
                    ROOT / "optimize_motion.py", case, "--device", "cuda", "--variant", "condmdi",
                    "--condmdi-branch", branch, "--condmdi-real"], e)
            queue.task_done()

    await asyncio.gather(*(worker(u) for u in GPU))
    STATE["state"] = "finished_needs_evaluation" if all(x["exit_code"] == 0 for x in STATE["jobs"].values()) else "failed"
    save()


if __name__ == "__main__":
    asyncio.run(main())
