"""Leased asynchronous pilot preparation. No RGB generation entry points."""
import asyncio
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

from scan_native import write, sha

ROOT = Path(__file__).resolve().parent
W = ROOT.parent
ENV = Path("/mnt/ojc/miniconda3/envs")
LOCKS = W / "gpu_leases_20261019"


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditioning", action="store_true")
    args = parser.parse_args()
    suffix = "conditioning" if args.conditioning else "gpu"
    status_path = ROOT / f"pilot_{suffix}_status.json"
    if status_path.exists():
        raise RuntimeError("Existing attempt requires explicit recovery")
    cpu = json.loads((ROOT / "pilot_prepare_status.json").read_text())
    rows = json.loads((ROOT / "pilot_manifest.json").read_text())["candidates"]
    assert all(cpu["jobs"][r["review_id"]]["state"] == "completed" and
               cpu["jobs"][r["review_id"]]["stage"] == "crop" for r in rows)
    (ROOT / "tmp").mkdir(exist_ok=True)
    (ROOT / "cache/torch_extensions").mkdir(parents=True, exist_ok=True)
    state = {"state": "running", "started": time.time(), "jobs": {}, "generation_started": False}
    frozen = {str(p): sha(p) for p in sorted((ROOT / "scripts").glob("*.py"))}
    write(ROOT / ("pilot_conditioning_code_frozen.json" if args.conditioning else "pilot_code_frozen.json"), frozen)
    queue = asyncio.Queue()
    for row in rows:
        queue.put_nowait(row)
    stages = [("interpolate", "4danyone", ["interpolate.py"]),
              ("seg_prepare", "4danyone", ["source_foreground.py", "prepare"]),
              ("segment", "4danyone-sam2", ["source_foreground.py", "segment"]),
              ("composite", "4danyone", ["source_foreground.py", "composite"]),
              ("motion", "4danyone", ["motion.py"])]
    if args.conditioning:
        previous = json.loads((ROOT / "pilot_gpu_status.json").read_text())
        assert all(previous["jobs"][r["review_id"]]["state"] == "completed" and
                   previous["jobs"][r["review_id"]]["stage"] == "motion" for r in rows)
        stages = [("conditioning", "4danyone", ["conditioning_prepare_only.py"])]

    async def worker(gpu):
        while not queue.empty():
            lease = (LOCKS / f"gpu{gpu}.lock").open("a")
            try:
                # Queue at the OS lock instead of repeatedly missing the very
                # short release/reacquire gap between tracking scenes.
                await asyncio.to_thread(fcntl.flock, lease, fcntl.LOCK_EX)
                query = await asyncio.create_subprocess_exec(
                    "nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits", stdout=asyncio.subprocess.PIPE)
                output, _ = await query.communicate()
                if query.returncode or int(output.strip()) < 16000 or shutil.disk_usage(ROOT).free < 8*2**30:
                    await asyncio.sleep(5)
                    continue
                try:
                    row = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                key, source = row["review_id"], row["sample_root"]
                runtime = W / "round2/moving61_runtime" if row["output_slots"] == 61 else W.parent / "repo"
                env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu),
                           OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                           PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                           FDANYONE_SAMPLE_ID=key, FDANYONE_SAMPLE_ROOT=source,
                           FDANYONE_MOVING_ROOT=source, FDANYONE_SOURCE_EXPERIMENT=source,
                           FDANYONE_WAYMO_ROOT=str(Path(source).parent / "conditions"),
                           FDANYONE_MOTION_VARIANT="calibrated_moving", FDANYONE_ROOT_VIEWS="24",
                           FDANYONE_RUNTIME=str(runtime), FDANYONE_FRAMES=str(row["output_slots"]),
                           TORCH_EXTENSIONS_DIR=str(ROOT / "cache/torch_extensions"),
                           TORCH_CUDA_ARCH_LIST="8.9", MAX_JOBS="2", TMPDIR=str(ROOT / "tmp"))
                env.pop("PYTHONPATH", None)
                env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
                for stage, environment, args in stages:
                    assert all(sha(Path(p)) == h for p, h in frozen.items()), "Frozen code changed"
                    command = [str(ENV / environment / "bin/python"), str(ROOT / "scripts" / args[0]), *args[1:]]
                    with (ROOT / "logs" / f"{key}_{stage}.log").open("x") as log:
                        child = await asyncio.create_subprocess_exec(*command, env=env, stdout=log, stderr=log)
                        state["jobs"][key] = {"stage": stage, "state": "running", "pid": child.pid,
                                              "gpu": gpu, "started": time.time()}
                        write(status_path, state)
                        code = await child.wait()
                    state["jobs"][key].update(exit_code=code, state="completed" if code == 0 else "failed")
                    write(status_path, state)
                    if code:
                        break
                queue.task_done()
            finally:
                fcntl.flock(lease, fcntl.LOCK_UN)
                lease.close()
    write(status_path, state)
    await asyncio.gather(*(worker(g) for g in range(8)))
    state.update(state="finished_requires_review", finished=time.time())
    write(status_path, state)


if __name__ == "__main__":
    asyncio.run(main())
