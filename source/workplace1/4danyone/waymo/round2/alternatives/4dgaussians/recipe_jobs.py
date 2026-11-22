"""One train/export/replay check, then two bounded independent backend recipes."""
import asyncio
from argparse import ArgumentParser
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

A = Path(__file__).resolve().parent
R = A.parents[1]
sys.path.insert(0, str(R / "scripts"))
from resource_lease import lease, wait_memory

STATE = {"state": "starting", "pid": os.getpid(), "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "jobs": {}}


def write():
    STATE["updated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    p = A / "queue.json"
    tmp = p.with_suffix(".writing")
    tmp.write_text(json.dumps(STATE, indent=2, allow_nan=False) + "\n")
    tmp.replace(p)


async def command(key, gpu, args, handles, environment):
    logpath = A / "logs" / f"{key}.log"
    command = ["/mnt/ojc/miniconda3/envs/4dgaussians/bin/python", *map(str, args)]
    with logpath.open("xb") as log:
        process = await asyncio.create_subprocess_exec(*command, cwd=A, env=environment,
            stdout=log, stderr=asyncio.subprocess.STDOUT, pass_fds=handles)
        STATE["jobs"][key] = {"state": "running", "pid": process.pid, "gpu": gpu, "command": command,
            "log": str(logpath), "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}
        write()
        code = await process.wait()
    (A / "logs" / f"{key}.exit").write_text(str(code) + "\n")
    STATE["jobs"][key].update(state="complete" if code == 0 else "failed", exit_code=code,
        ended_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
    write()
    if code:
        raise RuntimeError(f"{key} failed with exit{code}; inspect{logpath}")


async def pair(label, run, gpu, smoke=False, evaluation_only=False):
    STATE["jobs"][run] = {"state": "waiting_for_resources", "gpu": gpu}
    write()
    async with lease([gpu]) as handles:
        await wait_memory([gpu], 12000)
        assert shutil.disk_usage(A).free > 35 * 2**30
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
            CUDA_HOME="/mnt/ojc/miniconda3/envs/4dgaussians", PYTHONNOUSERSITE="1", OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", MPLCONFIGDIR=str(A / "cache/matplotlib"),
            TORCH_HOME=str(A / "cache/torch"))
        args = [A / "run_train.py", "--label", label, "--run", run] + (["--smoke"] if smoke else [])
        if evaluation_only:
            assert (A / "runs" / run / "complete.json").is_file()
            assert (A / "runs" / run / "reference_render.npz").is_file()
            evaluation_key = run + "_evaluation_r2"
        else:
            await command(run, gpu, args, handles, environment)
            evaluation_key = run + "_evaluation"
        await command(evaluation_key, gpu, [A / "evaluate.py", "--run", A / "runs" / run], handles, environment)


async def main():
    parser = ArgumentParser()
    parser.add_argument("--resume-smoke", action="store_true")
    args = parser.parse_args()
    assert not (A / "queue.json").exists(), "Do not restart completed/in-flight queue"
    (A / "logs").mkdir(exist_ok=True)
    write()
    await pair("batch_0084_twentyfour", "smoke0084_v3", 6, smoke=True, evaluation_only=args.resume_smoke)
    STATE["state"] = "two_complete_recipes_running"
    write()
    results = await asyncio.gather(pair("batch_0084_twentyfour", "0084_recipe30k_v1", 6),
        pair("walk063_rootlocal24", "walk063_recipe30k_v1", 7), return_exceptions=True)
    failures = [str(value) for value in results if isinstance(value, Exception)]
    STATE["state"] = "failed" if failures else "trained_replayed_evaluated_visual_review_pending"
    STATE["failures"] = failures
    write()
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as error:
        STATE.update(state="failed", exception=str(error))
        write()
        raise
