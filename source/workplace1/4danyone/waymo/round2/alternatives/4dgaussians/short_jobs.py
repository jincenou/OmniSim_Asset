"""Bounded short-sequence optimization, independent replay and FTGS comparisons."""
import asyncio
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

PATH = A / "short_queue.json"
STATE = {"state": "starting", "pid": os.getpid(), "jobs": {}}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def write():
    STATE["updated_at"] = now()
    temporary = PATH.with_suffix(".writing")
    temporary.write_text(json.dumps(STATE, indent=2, allow_nan=False) + "\n")
    temporary.replace(PATH)


async def command(key, gpu, python, arguments, handles, environment):
    assert not (A / "logs" / f"{key}.exit").exists()
    cmd = [python, *map(str, arguments)]
    logpath = A / "logs" / f"{key}.log"
    with logpath.open("xb") as log:
        process = await asyncio.create_subprocess_exec(*cmd, cwd=A, env=environment,
            stdout=log, stderr=asyncio.subprocess.STDOUT, pass_fds=handles)
        STATE["jobs"][key] = {"state": "running", "pid": process.pid, "physical_gpu": gpu,
            "command": cmd, "log": str(logpath), "started_at": now()}
        write()
        code = await process.wait()
    (A / "logs" / f"{key}.exit").write_text(str(code) + "\n")
    STATE["jobs"][key].update(state="complete" if code == 0 else "failed", exit_code=code, ended_at=now())
    write()
    if code:
        raise RuntimeError(f"{key}: exit{code}; inspect{logpath}")


async def one(label, prefix, gpu):
    STATE["jobs"][prefix] = {"state": "waiting_for_resources", "physical_gpu": gpu}
    write()
    async with lease([gpu]) as handles:
        await wait_memory([gpu], 12000)
        assert shutil.disk_usage(A).free > 35 * 2**30
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
            CUDA_HOME="/mnt/ojc/miniconda3/envs/4dgaussians", PYTHONNOUSERSITE="1", OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", MPLCONFIGDIR=str(A / "cache/matplotlib"),
            TORCH_HOME=str(A / "cache/torch"))
        python = "/mnt/ojc/miniconda3/envs/4dgaussians/bin/python"
        smoke = prefix + "_smoke_v1"
        full = prefix + "_recipe30k_v1"
        await command(smoke, gpu, python, [A / "run_train.py", "--label", label, "--run", smoke, "--smoke"], handles, environment)
        await command(smoke + "_evaluation", gpu, python, [A / "evaluate.py", "--run", A / "runs" / smoke], handles, environment)
        await command(full, gpu, python, [A / "run_train.py", "--label", label, "--run", full], handles, environment)
        await command(full + "_evaluation", gpu, python, [A / "evaluate.py", "--run", A / "runs" / full], handles, environment)
        ftgs = dict(environment, CUDA_HOME="/mnt/ojc/miniconda3/envs/asset_harvester_2026",
            TORCH_CUDA_ARCH_LIST="8.9", TORCH_EXTENSIONS_DIR=str(R.parent / "reconstruction/deps/torch_extensions"))
        ftgs["PATH"] = ftgs["CUDA_HOME"] + "/bin:" + ftgs["PATH"]
        ftgs.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        await command(full + "_comparison", gpu, "/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python",
            [R / "scripts/compare_alternative_backend.py", "--label", label, "--run", full], handles, ftgs)


async def main():
    assert not PATH.exists(), "Do not overwrite a live or completed queue"
    assert (A / "reports/runtime_preflight_short.json").is_file()
    STATE["started_at"] = now()
    write()
    results = await asyncio.gather(one("m02_short61_24", "m02_61", 4),
        one("r07_short61_24", "r07_61", 5), return_exceptions=True)
    errors = [str(value) for value in results if isinstance(value, Exception)]
    STATE.update(state="failed" if errors else "trained_replayed_evaluated_compared_visual_review_pending",
        failures=errors, ended_at=now())
    write()
    if errors:
        raise RuntimeError("; ".join(errors))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as error:
        STATE.update(state="failed", exception=str(error))
        write()
        raise
