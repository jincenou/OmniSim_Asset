"""Two geometry workers, followed by demos of the live short-sequence recipes."""
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

PATH = A / "demo_queue.json"
STATE = {"state": "starting", "pid": os.getpid(), "jobs": {}}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def write():
    STATE["updated_at"] = now()
    temporary = PATH.with_suffix(".writing")
    temporary.write_text(json.dumps(STATE, indent=2, allow_nan=False) + "\n")
    temporary.replace(PATH)


async def one(run, job, gpu, controller=None):
    source = A / "runs" / run / "evaluation/results.json"
    STATE["jobs"][run] = {"state": "waiting_for_verified_evaluation", "physical_gpu": gpu}
    write()
    while not source.exists():
        assert controller is not None
        command_path = Path(f"/proc/{controller}/cmdline")
        assert command_path.exists() and b"short_jobs.py" in command_path.read_bytes(), "Training controller no longer live"
        queue = json.loads((A / "short_queue.json").read_text())
        assert queue["state"] != "failed", "Short recipe queue failed; inspect its logs"
        await asyncio.sleep(30)
    assert json.loads(source.read_text())["state"] == "new_process_replayed_evaluated_decoded_visual_review_pending"
    STATE["jobs"][run]["state"] = "waiting_for_resources"
    write()
    async with lease([gpu]) as handles:
        await wait_memory([gpu], 8000)
        assert shutil.disk_usage(A).free > 35 * 2**30
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
            CUDA_HOME="/mnt/ojc/miniconda3/envs/4dgaussians", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
            MKL_NUM_THREADS="4", PYTHONNOUSERSITE="1", TORCH_HOME=str(A / "cache/torch"))
        cmd = ["/mnt/ojc/miniconda3/envs/4dgaussians/bin/python", str(A / "render_demos.py"),
            "--run", str(A / "runs" / run), "--job", str(job)]
        key = run + "_demo"
        logpath = A / "logs" / (key + ".log")
        with logpath.open("xb") as log:
            process = await asyncio.create_subprocess_exec(*cmd, cwd=A, env=environment,
                stdout=log, stderr=asyncio.subprocess.STDOUT, pass_fds=handles)
            STATE["jobs"][run].update(state="rendering", pid=process.pid, command=cmd, log=str(logpath), started_at=now())
            write()
            code = await process.wait()
        (A / "logs" / (key + ".exit")).write_text(str(code) + "\n")
        STATE["jobs"][run].update(state="rendered_decoded_review_pending" if code == 0 else "failed",
            exit_code=code, ended_at=now())
        write()
        if code:
            raise RuntimeError(f"{key}:exit{code};inspect{logpath}")


async def main():
    assert not PATH.exists(), "Do not overwrite a running or completed demo queue"
    controller = json.loads((A / "short_queue.json").read_text())["pid"]
    STATE.update(started_at=now(), short_controller_pid=controller)
    write()
    results = await asyncio.gather(
        one("walk063_recipe30k_v1", A / "demo_jobs/walk063.json", 6),
        one("0084_recipe30k_v1", R / "demo_jobs/batch_0084_twentyfour.json", 7),
        one("m02_61_recipe30k_v1", R / "demo_jobs/m02_short61_24.json", 6, controller),
        one("r07_61_recipe30k_v1", R / "demo_jobs/r07_short61_24.json", 7, controller), return_exceptions=True)
    errors = [str(value) for value in results if isinstance(value, Exception)]
    STATE.update(state="failed" if errors else "four_demos_rendered_decoded_visual_review_pending",
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
