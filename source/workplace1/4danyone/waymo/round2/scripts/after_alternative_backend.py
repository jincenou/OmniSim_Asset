"""Wait on the live recipe controller; compare each verified completed result."""
import asyncio
import json
import os
from pathlib import Path
import shutil
from after_batch import ROUND, W, now, write
from resource_lease import lease, wait_memory

A = ROUND / "alternatives/4dgaussians"
STATE = {"state": "waiting_for_existing_recipes", "pid": os.getpid(), "started_at": now(), "jobs": {}}
PATH = ROUND / "alternative_comparisons.json"


async def one(label, run, gpu, controller):
    source = A / "runs" / run / "evaluation/results.json"
    while not source.exists():
        cmdline = Path(f"/proc/{controller}/cmdline")
        if not cmdline.exists() or b"recipe_jobs.py" not in cmdline.read_bytes():
            raise RuntimeError(f"Recipe controller{controller} no longer live; no validated{run} result")
        await asyncio.sleep(30)
    assert json.loads(source.read_text())["state"] == "new_process_replayed_evaluated_decoded_visual_review_pending"
    async with lease([gpu]) as handles:
        await wait_memory([gpu], 6000)
        assert shutil.disk_usage(ROUND).free > 35 * 2**30
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
            CUDA_HOME="/mnt/ojc/miniconda3/envs/asset_harvester_2026", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
            TORCH_CUDA_ARCH_LIST="8.9", TORCH_EXTENSIONS_DIR=str(W / "reconstruction/deps/torch_extensions"))
        env["PATH"] = env["CUDA_HOME"] + "/bin:" + env["PATH"]
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        cmd = ["/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python", str(ROUND / "scripts/compare_alternative_backend.py"),
            "--label", label, "--run", run]
        with (ROUND / "logs" / f"alternative_compare_{run}.log").open("xb") as log:
            child = await asyncio.create_subprocess_exec(*cmd, cwd=ROUND / "backend", env=env,
                stdout=log, stderr=asyncio.subprocess.STDOUT, pass_fds=handles)
            STATE["jobs"][run] = {"state": "running", "gpu": gpu, "pid": child.pid, "command": cmd}
            write(PATH, STATE)
            code = await child.wait()
        (ROUND / "logs" / f"alternative_compare_{run}.exit").write_text(str(code) + "\n")
        STATE["jobs"][run].update(state="rendered_decoded_review_pending" if code == 0 else "failed", exit_code=code)
        write(PATH, STATE)
        if code:
            raise RuntimeError(f"Comparison{run} failed: exit{code}")


async def main():
    assert not PATH.exists()
    controller = json.loads((A / "queue.json").read_text())["pid"]
    STATE["recipe_controller_pid"] = controller
    write(PATH, STATE)
    results = await asyncio.gather(one("batch_0084_twentyfour", "0084_recipe30k_v1", 4, controller),
        one("walk063_rootlocal24", "walk063_recipe30k_v1", 5, controller), return_exceptions=True)
    errors = [str(value) for value in results if isinstance(value, Exception)]
    STATE.update(state="failed" if errors else "rendered_decoded_visual_review_pending", failures=errors, ended_at=now())
    write(PATH, STATE)
    if errors:
        raise RuntimeError("; ".join(errors))


if __name__ == "__main__":
    asyncio.run(main())
