"""Complete N06/N07/N08/N11 using newly authorized cards 0-7.

Per-device leases are shared with every existing queue. Existing prefix and
training receipts are adopted, not regenerated. No recipe or actor is added.
"""
import asyncio
from contextlib import asynccontextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil

import controller_v3 as queue

R = Path(__file__).resolve().parents[1]
LABELS = ("N06", "N07", "N08", "N11")
STATE = R / "ALL8_STATUS.json"
BASE_STAGE = queue.stage
BASE_WRITE = queue.write
MAP = {"full_masks.py": "full_masks_all8.py", "train_ftgs.py": "train_ftgs_all8.py",
       "train_4dgs.py": "train_4dgs_all8.py", "evaluate_branch.py": "evaluate_branch_all8.py"}


def persist():
    queue.state.update(updated_at=queue.now(), disk_free_bytes=shutil.disk_usage(R).free)
    BASE_WRITE(STATE, queue.state)


def write(path, data):
    BASE_WRITE(path, data)
    if path.parent == R / "logs" and data.get("pid") and data.get("exit_code") is not None:
        archive = R / "logs/actual_receipts" / f"{path.stem}_pid{data['pid']}_exit{data['exit_code']}_all8.json"
        if not archive.exists():
            BASE_WRITE(archive, {"source": STATE.name, "job": path.stem, "actual_receipt": data})


@asynccontextmanager
async def lease(ids):
    assert ids and set(ids) <= set(range(8))
    handles = [(queue.OLD / "controller.lock").open("a"), (queue.W / "batch100/controller.lock").open("a")]
    handles.extend((queue.OLD / "gpu_locks" / f"gpu{gpu}.lock").open("a") for gpu in sorted(ids))
    try:
        while True:
            acquired = []
            try:
                for index, handle in enumerate(handles):
                    fcntl.flock(handle, (fcntl.LOCK_SH if index < 2 else fcntl.LOCK_EX) | fcntl.LOCK_NB)
                    acquired.append(handle)
            except BlockingIOError:
                for handle in acquired:
                    fcntl.flock(handle, fcntl.LOCK_UN)
                await asyncio.sleep(5)
            else:
                break
        yield tuple(handle.fileno() for handle in handles)
    finally:
        for handle in handles:
            handle.close()


async def choose_gpu(minimum):
    while True:
        process = await asyncio.create_subprocess_exec("nvidia-smi", "--query-gpu=index,memory.free",
                    "--format=csv,noheader,nounits", stdout=asyncio.subprocess.PIPE)
        output, _ = await process.communicate()
        assert process.returncode == 0
        candidates = [tuple(map(int, row.split(b","))) for row in output.splitlines()]
        for gpu, free in sorted(candidates, key=lambda item: (-item[1], item[0])):
            if free < minimum:
                continue
            with (queue.OLD / "gpu_locks" / f"gpu{gpu}.lock").open("a") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
            return gpu
        await asyncio.sleep(10)


async def stage(label, name, kind, script, *args, **kwargs):
    assert label in LABELS
    return await BASE_STAGE(label, name, kind, MAP.get(script, script), *args, **kwargs)


async def main():
    lock = (R / "all8_queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not STATE.exists(), "No silent restart; inspect registered receipts first"
    queue.state = {"state": "all_eight_devices_authorized_remaining_tracks_active", "controller_pid": os.getpid(),
                   "allowed_gpu_ids": list(range(8)), "owned_tracks": list(LABELS),
                   "jobs": {}, "samples": {}, "production_approved": False}
    queue.persist, queue.write, queue.stage = persist, write, stage
    queue.lease, queue.choose_gpu = lease, choose_gpu
    queue.MATERIALIZE = asyncio.Semaphore(4)
    queue.BACK_SEM = asyncio.Semaphore(4)
    registered = {case["sample_id"]: case for case in queue.selection["main"] + queue.selection["reserve"]}
    cases = [registered[label] for label in LABELS]
    for case in cases:
        label = case["sample_id"]
        proof = R / "samples" / label / "generation/research/generated_validation.json"
        assert json.loads(proof.read_text())["state"] == "all24_videos_decoded_same_retained_cameras"
        queue.state["samples"][label] = {"state": "shared_generation_verified_pending_reconstruction"}
        for branch in ("ftgs_base5k", "ftgs_opt15k", "4dgs_grid25", "4dgs_grid61"):
            queue.state["jobs"][f"{label}/{branch}"] = {"state": "waiting_shared_inputs_or_device_lease"}
    persist()
    await asyncio.gather(*map(queue.reconstruct, cases))
    queue.state.update(state="all8_supplement_registered_tasks_terminal_review_pending", ended_at=queue.now())
    persist()


if __name__ == "__main__":
    asyncio.run(main())
