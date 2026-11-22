"""Two controlled alpha-objective recipes sharing frozen short-sequence inputs."""
import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

A = Path(__file__).resolve().parent
import short_jobs as controller


async def one(label, prefix, gpu):
    controller.STATE["jobs"][prefix] = {"state": "waiting_for_resources", "physical_gpu": gpu}
    controller.write()
    async with controller.lease([gpu]) as handles:
        await controller.wait_memory([gpu], 12000)
        assert controller.shutil.disk_usage(A).free > 35 * 2**30
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
            CUDA_HOME="/mnt/ojc/miniconda3/envs/4dgaussians", PYTHONNOUSERSITE="1", OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", TORCH_HOME=str(A / "cache/torch"))
        python = "/mnt/ojc/miniconda3/envs/4dgaussians/bin/python"
        smoke, full = prefix + "_smoke_v1", prefix + "_recipe30k_v1"
        await controller.command(smoke, gpu, python, [A / "run_train.py", "--label", label, "--run", smoke,
            "--smoke", "--objective", "alpha_l1"], handles, env)
        await controller.command(smoke + "_evaluation", gpu, python, [A / "evaluate.py", "--run", A / "runs" / smoke], handles, env)
        await controller.command(full, gpu, python, [A / "run_train.py", "--label", label, "--run", full,
            "--objective", "alpha_l1"], handles, env)
        await controller.command(full + "_evaluation", gpu, python, [A / "evaluate.py", "--run", A / "runs" / full], handles, env)
        await controller.command(full + "_demo", gpu, python, [A / "render_demos.py", "--run", A / "runs" / full,
            "--job", controller.R / "demo_jobs" / (label + ".json")], handles, env)


async def main():
    controller.PATH = A / "alpha_queue.json"
    assert not controller.PATH.exists()
    controller.STATE.update(state="two_alpha_objective_recipes", started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
    controller.write()
    results = await asyncio.gather(one("m02_short61_24", "m02_61_alpha", 6),
        one("r07_short61_24", "r07_61_alpha", 7), return_exceptions=True)
    errors = [str(value) for value in results if isinstance(value, Exception)]
    controller.STATE.update(state="failed" if errors else "alpha_recipes_replayed_evaluated_decoded_pair_comparison_pending", failures=errors)
    controller.write()
    if errors:
        raise RuntimeError("; ".join(errors))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as error:
        controller.STATE.update(state="failed", exception=str(error))
        controller.write()
        raise
