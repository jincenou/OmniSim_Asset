"""CPU decoding does not wait for a 35-GiB GPU admission gate."""
import asyncio
import json
import os
from pathlib import Path
import time

G = Path(__file__).resolve().parent
B = G / "generation_dev6"
PY = "/mnt/ojc/miniconda3/envs/4danyone/bin/python"


def save(state):
    temp = B / "cpu_validation.writing"
    temp.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
    temp.replace(B / "cpu_validation.json")


async def main():
    assert not (B / "cpu_validation.json").exists()
    subjects = json.loads((B / "PROTOCOL.json").read_text())["subjects"]
    pairs = [(sample, label) for sample in subjects for label in ("G1", "G2")]
    tasks = {}
    sem = asyncio.Semaphore(2)
    state = {"state": "watching_completed_generation", "jobs": {},
             "reason": "Original orchestration applies a GPU memory gate even to CPU validation; decode independent completed output without changing generation"}
    async def validate(sample, label):
        key = f"{sample}_{label}"
        async with sem:
            root = B / sample / label
            if (root / "validation.json").exists():
                state["jobs"][key] = {"state": "already_validated"}
                save(state)
                return
            with (B / "logs" / f"{key}_cpu_validate.log").open("x") as log:
                proc = await asyncio.create_subprocess_exec(
                    PY, str(G / "validate_dev6_generation.py"), sample, label,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2"),
                    stdout=log, stderr=log)
                state["jobs"][key] = dict(state="running", pid=proc.pid)
                save(state)
                rc = await proc.wait()
            state["jobs"][key].update(state="validated" if rc == 0 else "failed", exit_code=rc)
            save(state)
    while True:
        for sample, label in pairs:
            root = B / sample / label
            key = f"{sample}_{label}"
            if (root / "validation.json").exists():
                state["jobs"].setdefault(key, {"state": "already_validated"})
            elif key not in tasks and (root / "status.json").exists():
                status = json.loads((root / "status.json").read_text())
                if status["state"] == "generated_validation_pending":
                    tasks[key] = asyncio.create_task(validate(sample, label))
        if all((B / sample / label / "validation.json").exists() for sample, label in pairs):
            break
        first = json.loads((B / "status.json").read_text())["state"]
        repair = json.loads((B / "P04_geometry_repair/status.json").read_text())["state"]
        terminal = {"queue_failed_preserved_results", "generated_validated_needs_visual_review", "controller_failed", "repair_failed"}
        if first in terminal and repair in terminal:
            await asyncio.gather(*tasks.values())
            break
        state["updated_at"] = time.time()
        save(state)
        await asyncio.sleep(20)
    await asyncio.gather(*tasks.values())
    state["state"] = "completed_available_outputs"
    state["unvalidated"] = [f"{s}_{b}" for s, b in pairs if not (B / s / b / "validation.json").exists()]
    save(state)


if __name__ == "__main__":
    asyncio.run(main())
