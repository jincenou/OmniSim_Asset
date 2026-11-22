"""Evaluate saved budget snapshots of four existing models without retraining."""
import asyncio
import os
from pathlib import Path
import short_jobs as controller
from additional_jobs import environment

A = Path(__file__).resolve().parent


async def one(run, gpu):
    async with controller.lease([gpu]) as handles:
        await controller.wait_memory([gpu], 8000)
        assert controller.shutil.disk_usage(A).free > 35 * 2**30
        env = environment(gpu)
        python = "/mnt/ojc/miniconda3/envs/4dgaussians/bin/python"
        await controller.command(run + "_eval5000", gpu, python,
            [A / "evaluate.py", "--run", A / "runs" / run, "--iteration", 5000], handles, env)
        await controller.command(run + "_eval15000", gpu, python,
            [A / "evaluate.py", "--run", A / "runs" / run, "--iteration", 15000], handles, env)


async def main():
    controller.PATH = A / "budget_queue.json"
    assert not controller.PATH.exists()
    controller.STATE.update(state="evaluating_existing_snapshots_no_training", started_at=controller.now())
    controller.write()
    values = await asyncio.gather(one("m02_61_recipe30k_v1", 4), one("r07_61_recipe30k_v1", 5),
        one("walkturn0021_recipe30k_v1", 6), one("phone_recipe30k_v1", 7), return_exceptions=True)
    errors = [str(value) for value in values if isinstance(value, Exception)]
    controller.STATE.update(state="failed" if errors else "eight_snapshot_evaluations_decoded_no_training_review_pending", failures=errors)
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
