"""Fill unused device leases with the final two already-frozen actors.

The first all-eight queue already has four long reconstructions running.
This queue admits N08/N11 now, sharing all receipts and GPU locks, so the
last card need not wait for an unrelated actor to finish its whole recipe.
"""
import asyncio
import fcntl
import os

import all8_queue as adapter

R = adapter.R
LABELS = ("N08", "N11")


async def main():
    lock = (R / "spare_queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    adapter.STATE = R / "SPARE_STATUS.json"
    adapter.LABELS = LABELS
    assert not adapter.STATE.exists(), "Inspect existing receipts before any restart"
    queue = adapter.queue
    queue.state = {"state": "spare_device_queue_active", "controller_pid": os.getpid(),
                   "allowed_gpu_ids": list(range(8)), "owned_tracks": list(LABELS),
                   "jobs": {}, "samples": {}, "production_approved": False,
                   "scope": "Frozen N08/N11 only; existing queues adopt these shared stage receipts. No extra identities or recipes."}
    queue.persist, queue.write, queue.stage = adapter.persist, adapter.write, adapter.stage
    queue.lease, queue.choose_gpu = adapter.lease, adapter.choose_gpu
    queue.MATERIALIZE = asyncio.Semaphore(2)
    queue.BACK_SEM = asyncio.Semaphore(4)
    registered = {case["sample_id"]: case for case in queue.selection["main"] + queue.selection["reserve"]}
    cases = [registered[label] for label in LABELS]
    for case in cases:
        label = case["sample_id"]
        assert (R / "research" / f"{label}-mask-complete.json").exists()
        assert (R / "four_dg/data" / label / "fdanyone.json").exists()
        queue.state["samples"][label] = {"state": "shared_inputs_verified_pending_reconstruction"}
        for branch in ("ftgs_base5k", "ftgs_opt15k", "4dgs_grid25", "4dgs_grid61"):
            queue.state["jobs"][f"{label}/{branch}"] = {"state": "waiting_shared_stage_receipt_or_device_lease"}
    adapter.persist()
    await asyncio.gather(*map(queue.reconstruct, cases))
    queue.state.update(state="spare_registered_tasks_terminal_review_pending", ended_at=queue.now())
    adapter.persist()


if __name__ == "__main__":
    asyncio.run(main())
