"""Two frozen-recipe checks on standing and walking/turning tracks."""
import asyncio
import os
from pathlib import Path
import short_jobs as controller

A = Path(__file__).resolve().parent
R = A.parents[1]


def environment(gpu, ftgs=False):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONNOUSERSITE="1",
        OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", TORCH_HOME=str(A / "cache/torch"))
    env["CUDA_HOME"] = "/mnt/ojc/miniconda3/envs/asset_harvester_2026" if ftgs else "/mnt/ojc/miniconda3/envs/4dgaussians"
    if ftgs:
        env.update(TORCH_CUDA_ARCH_LIST="8.9", TORCH_EXTENSIONS_DIR=str(R.parent / "reconstruction/deps/torch_extensions"))
        env["PATH"] = env["CUDA_HOME"] + "/bin:" + env["PATH"]
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    return env


async def export(label, data, gpu):
    async with controller.lease([gpu]) as handles:
        assert controller.shutil.disk_usage(A).free > 35 * 2**30
        await controller.command("additional_export_" + label, gpu, "/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python",
            [A / "export_data.py", "--label", label, "--data", data], handles, environment(gpu, True))


async def one(label, prefix, gpu):
    async with controller.lease([gpu]) as handles:
        await controller.wait_memory([gpu], 12000)
        assert controller.shutil.disk_usage(A).free > 35 * 2**30
        python = "/mnt/ojc/miniconda3/envs/4dgaussians/bin/python"
        run = prefix + "_recipe30k_v1"
        await controller.command(run, gpu, python, [A / "run_train.py", "--label", label, "--run", run], handles, environment(gpu))
        await controller.command(run + "_evaluation", gpu, python, [A / "evaluate.py", "--run", A / "runs" / run], handles, environment(gpu))
        await controller.command(run + "_comparison", gpu, "/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python",
            [R / "scripts/compare_alternative_backend.py", "--label", label, "--run", run], handles, environment(gpu, True))
        await controller.command(run + "_demo", gpu, python, [A / "render_demos.py", "--run", A / "runs" / run,
            "--job", R / "demo_jobs" / (label + ".json")], handles, environment(gpu))


async def main():
    controller.PATH = A / "additional_queue.json"
    assert not controller.PATH.exists()
    controller.STATE.update(state="preparing_two_independent_frozen_recipes", started_at=controller.now())
    controller.write()
    values = await asyncio.gather(export("phone_full", R.parent / "reconstruction/data/phone_full", 4),
        export("batch_0021_rootlocal24", R / "reconstruction/data/batch_0021_rootlocal24", 5), return_exceptions=True)
    errors = [str(item) for item in values if isinstance(item, Exception)]
    if errors:
        raise RuntimeError("; ".join(errors))
    async with controller.lease([4]) as handles:
        await controller.wait_memory([4], 8000)
        await controller.command("additional_preflight", 4, "/mnt/ojc/miniconda3/envs/4dgaussians/bin/python",
            [A / "preflight.py", "--labels", "phone_full", "batch_0021_rootlocal24", "--output",
                A / "reports/runtime_preflight_additional.json"], handles, environment(4))
    values = await asyncio.gather(one("phone_full", "phone", 4), one("batch_0021_rootlocal24", "walkturn0021", 5), return_exceptions=True)
    errors = [str(item) for item in values if isinstance(item, Exception)]
    controller.STATE.update(state="failed" if errors else "additional_recipes_replayed_evaluated_decoded_review_pending", failures=errors)
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
