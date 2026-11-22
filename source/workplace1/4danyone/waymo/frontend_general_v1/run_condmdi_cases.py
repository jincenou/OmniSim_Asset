import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "text_motion"
PY = "/mnt/ojc/miniconda3/envs/condmdi/bin/python"
GPU = ["0db9aa59-0293-a3ea-06a0-395913880fbd", "4e635188-f5de-33f9-6ecb-53555a0a7c9d",
       "e3be373d-e235-e8d2-76bf-3559bea96b58", "16387ea4-8329-06c9-0c83-c18c0e71310c",
       "f9cfc0e8-a5ac-7168-2700-4b52fee1e21d", "85bb9093-960d-9196-9cf8-cdecf8fd0765",
       "cd4b2342-fba4-2b42-beb9-60b698debc2d", "32040e1a-df6d-0776-5420-fae47bc11248"]
STATE = {"state": "preparing", "jobs": {}}


def save():
    temp = OUT / "cases_status.writing"
    temp.write_text(json.dumps(STATE, indent=2) + "\n")
    temp.replace(OUT / "cases_status.json")


async def command(key, argv, env):
    with (OUT / "logs" / (key + ".log")).open("x") as log:
        p = await asyncio.create_subprocess_exec(PY, *map(str, argv), env=env, stdout=log, stderr=log)
        STATE["jobs"][key] = dict(state="running", pid=p.pid, started=time.time())
        save()
        rc = await p.wait()
        STATE["jobs"][key].update(state="finished" if rc == 0 else "failed", exit_code=rc, finished=time.time())
        save()
        return rc


async def main():
    protocol = json.loads((OUT / "PROTOCOL.json").read_text())
    assert json.loads((OUT / "smoke_status.json").read_text())["exit_code"] == 0
    (OUT / "logs").mkdir(exist_ok=True)
    assert not (OUT / "cases_status.json").exists()
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    sem = asyncio.Semaphore(2)
    async def prepare(case):
        async with sem:
            return await command(case + "_prepare", [ROOT / "prepare_condmdi_case.py", case], env)
    prepared = await asyncio.gather(*(prepare(c) for c in protocol["cases"]))
    if any(prepared):
        STATE["state"] = "preparation_failed"
        save()
        return
    STATE["state"] = "sampling"
    queue = asyncio.Queue()
    [queue.put_nowait((c, b)) for c in protocol["cases"] for b in ["no_text", "text"]]
    async def worker(uuid):
        worker_env = dict(env, CUDA_VISIBLE_DEVICES="GPU-" + uuid, FD_EXPECTED_GPU_UUID="GPU-" + uuid)
        while not queue.empty():
            case, branch = queue.get_nowait()
            await command(case + "_" + branch, [ROOT / "sample_condmdi_case.py", case, branch], worker_env)
            queue.task_done()
    await asyncio.gather(*(worker(u) for u in GPU))
    STATE["state"] = "finished_needs_retargeting" if all(j["exit_code"] == 0 for j in STATE["jobs"].values()) else "failed"
    save()


if __name__ == "__main__":
    asyncio.run(main())
