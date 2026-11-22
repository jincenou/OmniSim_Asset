"""Move only not-yet-started short demos to GPUs freed by official recipes."""
import asyncio
from pathlib import Path
import demo_jobs as controller

A = Path(__file__).resolve().parent


async def main():
    controller.PATH = A / "short_demo_queue.json"
    assert not controller.PATH.exists()
    controller.STATE.update(state="short_demos_on_released_training_devices", started_at=controller.now())
    controller.write()
    values = await asyncio.gather(
        controller.one("m02_61_recipe30k_v1", controller.R / "demo_jobs/m02_short61_24.json", 4),
        controller.one("r07_61_recipe30k_v1", controller.R / "demo_jobs/r07_short61_24.json", 5), return_exceptions=True)
    errors = [str(value) for value in values if isinstance(value, Exception)]
    controller.STATE.update(state="failed" if errors else "short_demos_rendered_decoded_visual_review_pending", failures=errors)
    controller.write()
    if errors:
        raise RuntimeError("; ".join(errors))


if __name__ == "__main__":
    asyncio.run(main())
