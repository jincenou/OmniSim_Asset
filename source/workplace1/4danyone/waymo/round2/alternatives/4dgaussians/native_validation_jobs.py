"""Frozen grid61 recipe on two identities unused for choosing temporal settings."""
import asyncio
from pathlib import Path
import temporal_recipe_jobs as controller

A = Path(__file__).resolve().parent


async def main():
    controller.STATE_PATH = A / 'native_validation_queue.json'
    assert not controller.STATE_PATH.exists(), 'Do not restart running or completed validation'
    controller.STATE['state'] = 'held_recipe_121frame_smoke'
    controller.write()
    await controller.recipe('phone_full', 'phone_grid61_smoke_v1', 'grid61', 4, True)
    controller.STATE['state'] = 'two_independent_121frame_grid61_recipes_running'
    controller.write()
    results = await asyncio.gather(
        controller.recipe('phone_full', 'phone_grid61_recipe30k_v1', 'grid61', 4),
        controller.recipe('batch_0021_rootlocal24', 'walkturn0021_grid61_recipe30k_v1', 'grid61', 5),
        return_exceptions=True)
    await controller.gate_results(results, 'two_independent_recipes_trained_replayed_profiled_rendered_compared_review_pending')


if __name__ == '__main__':
    asyncio.run(main())
