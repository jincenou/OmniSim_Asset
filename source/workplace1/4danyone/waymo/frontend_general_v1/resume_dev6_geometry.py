"""Resume the isolated geometry failure, preserving the first queue's exit evidence."""
import asyncio
import hashlib
import json
import os
import time
import run_dev6_generation as queue

G, B = queue.G, queue.B
REPAIR = B / "P04_geometry_repair"


def save():
    queue.STATE["updated_at"] = time.time()
    temp = REPAIR / "status.writing"
    temp.write_text(json.dumps(queue.STATE, indent=2, allow_nan=False) + "\n")
    temp.replace(REPAIR / "status.json")


async def main():
    assert not (REPAIR / "status.json").exists()
    plan = json.loads((REPAIR / "PLAN.json").read_text())
    for path, digest in plan["code_sha256"].items():
        assert hashlib.sha256(queue.Path(path).read_bytes()).hexdigest() == digest
    query = await asyncio.create_subprocess_exec(
        "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader",
        stdout=asyncio.subprocess.PIPE)
    stdout, _ = await query.communicate()
    assert query.returncode == 0
    queue.UUIDS.update({int(line.split(", ")[0]): line.split(", ")[1] for line in stdout.decode().splitlines()})
    for gpu in range(8):
        queue.AVAILABLE.put_nowait(gpu)
    # Geometry preparation has measured ~0.5 GiB usage, may share a GPU with diffusion.
    # Subsequent diffusion still takes the ordinary exclusive per-task lease and 35 GiB gate.
    gpu = 5
    while True:
        check = await asyncio.create_subprocess_exec(
            "nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE)
        free, _ = await check.communicate()
        assert check.returncode == 0
        if int(free.strip()) >= 8192:
            break
        queue.STATE["state"] = "waiting_shared_geometry_memory"
        save()
        await asyncio.sleep(15)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(
        CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu),
        FD_EXPECTED_GPU_UUID=queue.UUIDS[gpu], FDANYONE_RUNTIME=str(G.parent.parent / "repo"),
        OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        TORCH_EXTENSIONS_DIR=str(G.parent / "scene20/r20261019/cache/torch_extensions"),
        TORCH_CUDA_ARCH_LIST="8.9", MAX_JOBS="2", TMPDIR=str(B / "tmp"))
    with (REPAIR / "prepare.log").open("x") as log:
        start = time.time()
        proc = await asyncio.create_subprocess_exec(queue.PY, str(G / "prepare_dev6_analytic_maps.py"),
                                                   "P04", env=env, stdout=log, stderr=log)
        queue.STATE.update(state="preparing_geometry", geometry_pid=proc.pid, geometry_gpu=gpu)
        save()
        rc = await proc.wait()
    queue.STATE["geometry_receipt"] = dict(exit_code=rc, seconds=time.time() - start)
    save()
    if rc:
        raise RuntimeError(f"Analytic geometry preparation exited {rc}")
    queue.STATE["state"] = "generating"
    save()
    await queue.finish_pair("P04")
    queue.STATE["state"] = "generated_validated_needs_visual_review"
    save()


if __name__ == "__main__":
    queue.save = save
    try:
        asyncio.run(main())
    except BaseException as error:
        queue.STATE.update(state="repair_failed", error=repr(error))
        save()
        raise
