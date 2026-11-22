"""Resume only missing phone comparison/demo; never restart its completed optimization."""
import asyncio
import json
from pathlib import Path
import short_jobs as controller
from additional_jobs import environment

A = Path(__file__).resolve().parent


async def main():
    controller.PATH = A / "phone_recovery_queue.json"
    assert not controller.PATH.exists()
    root = A / "runs/phone_recipe30k_v1"
    assert json.loads((root / "evaluation/results.json").read_text())["state"] == "new_process_replayed_evaluated_decoded_visual_review_pending"
    assert not (root / "ftgs_comparison").exists() and not (root / "demo").exists()
    controller.STATE.update(state="resuming_completed_model_postprocessing", started_at=controller.now())
    controller.write()
    async with controller.lease([4]) as handles:
        await controller.wait_memory([4], 8000)
        await controller.command("phone_recipe30k_v1_comparison_r2", 4, "/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python",
            [controller.R / "scripts/compare_alternative_backend.py", "--label", "phone_full", "--run", root.name], handles, environment(4, True))
        await controller.command("phone_recipe30k_v1_demo", 4, "/mnt/ojc/miniconda3/envs/4dgaussians/bin/python",
            [A / "render_demos.py", "--run", root, "--job", controller.R / "demo_jobs/phone_full.json"], handles, environment(4))
    controller.STATE["state"] = "phone_comparison_and_demo_recovered_decoded_visual_review_pending"
    controller.write()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as error:
        controller.STATE.update(state="failed", exception=str(error))
        controller.write()
        raise
