"""Wait for assets, run public checkpoints, and validate visualization artifacts."""
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/mnt/ojc/miniconda3/envs/4danyone/bin/python"


def status(stage, state, **extra):
    data = {"updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "scope": "public_checkpoint_multiview_video_only", "training": False, "freetimegs": "deferred", "stage": stage, "state": state, **extra}
    temporary = ROOT / "status.json.tmp"
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(ROOT / "status.json")
    print(json.dumps(data, ensure_ascii=False), flush=True)


async def wait_ready(name):
    while not (ROOT / "logs" / f"{name}.complete").is_file():
        process = await asyncio.create_subprocess_exec("tmux", "-L", "fdanyone", "has-session", "-t", name, stderr=asyncio.subprocess.DEVNULL)
        if await process.wait():
            log_name = "download-http.log" if name == "download" else f"{name}.log"
            raise RuntimeError(f"{name} exited without completion marker; inspect logs/{log_name}")
        await asyncio.sleep(60)


async def execute(stage, *command):
    status(stage, "running", command=command)
    with (ROOT / "logs" / f"{stage}.log").open("w") as log:
        process = await asyncio.create_subprocess_exec(*command, stdout=log, stderr=asyncio.subprocess.STDOUT, cwd=ROOT)
        status(stage, "running", pid=process.pid, command=command)
        code = await process.wait()
    if code:
        raise RuntimeError(f"{stage} failed with exit code {code}; see logs/{stage}.log")


async def run_mode(mode):
    physical_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    selected = physical_ids[:1] if mode == "smoke" else physical_ids
    status("shared_gpu_start", "running", next_mode=mode, physical_gpu_ids=selected, sharing="explicitly_authorized")
    await execute(mode, "bash", str(ROOT / "scripts/run.sh"), mode)
    await execute(f"{mode}-preview", PYTHON, str(ROOT / "scripts/visualize.py"), str(ROOT / "outputs" / mode))


async def main():
    status("dependencies_and_assets", "running")
    try:
        await asyncio.gather(wait_ready("setup"), wait_ready("download"))
        await run_mode("smoke")
        await run_mode("turbo")
        await run_mode("base")
        status("videos_and_previews", "validated", outputs=["outputs/smoke/preview", "outputs/turbo/preview", "outputs/base/preview"], visual_review="pending_human_or_assistant_inspection")
    except Exception as error:
        status("pipeline", "failed", error=str(error))
        raise


if __name__ == "__main__":
    asyncio.run(main())
