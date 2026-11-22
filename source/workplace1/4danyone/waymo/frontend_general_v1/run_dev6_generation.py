"""Named-tmux bounded async queue, explicit GPU leases, no baseline overwrite."""
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

G = Path(__file__).resolve().parent
W = G.parent
B = G / "generation_dev6"
PY = "/mnt/ojc/miniconda3/envs/4danyone/bin/python"
STATE = {"state": "running", "jobs": {}, "quality_approved": False}
AVAILABLE = asyncio.Queue()
UUIDS = {}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save():
    STATE["updated_at"] = time.time()
    temp = B / "status.writing"
    temp.write_text(json.dumps(STATE, indent=2, allow_nan=False) + "\n")
    temp.replace(B / "status.json")


async def execute(sample, stage, script, args=(), label=None):
    key = f"{sample}_{stage}" + (f"_{label}" if label else "")
    protocol = json.loads((B / "PROTOCOL.json").read_text())
    frozen = json.loads((B / "code_freeze.json").read_text())
    for path, digest in frozen.items():
        assert sha(Path(path)) == digest, f"Changed frozen code: {path}"
    gpu = await AVAILABLE.get()
    lease = (W / "gpu_leases_20261019" / f"gpu{gpu}.lock").open("a")
    try:
        while True:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                await asyncio.sleep(15)
                continue
            query = await asyncio.create_subprocess_exec(
                "nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free,uuid",
                "--format=csv,noheader,nounits", stdout=asyncio.subprocess.PIPE)
            stdout, _ = await query.communicate()
            assert query.returncode == 0
            free, uuid = stdout.decode().strip().split(", ")
            assert uuid == UUIDS[gpu]
            if int(free) >= 35000 and shutil.disk_usage(B).free >= 8 * 2**30:
                break
            fcntl.flock(lease, fcntl.LOCK_UN)
            STATE["jobs"][key] = dict(state="waiting_resources", gpu=gpu)
            save()
            await asyncio.sleep(15)
        count = protocol["frames"][sample]
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        env.update(
            CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu),
            FD_EXPECTED_GPU_UUID=UUIDS[gpu], FDANYONE_FRAMES=str(count),
            FDANYONE_RUNTIME=str(W / "round2/moving61_runtime" if count == 61 else W.parent / "repo"),
            FDANYONE_RETAINED_VIEWS="24", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            TORCH_EXTENSIONS_DIR=str(W / "scene20/r20261019/cache/torch_extensions"),
            TORCH_CUDA_ARCH_LIST="8.9", MAX_JOBS="2", TMPDIR=str(B / "tmp"))
        if label:
            env["FDANYONE_WAYMO_ROOT"] = str(B / sample / label)
        with (B / "logs" / f"{key}.log").open("x") as log:
            start = time.time()
            proc = await asyncio.create_subprocess_exec(PY, str(script), *args,
                                                       env=env, stdout=log, stderr=log)
            STATE["jobs"][key] = dict(state="running", gpu=gpu, uuid=UUIDS[gpu],
                                      pid=proc.pid, started=start)
            save()
            rc = await proc.wait()
        STATE["jobs"][key].update(state="completed" if rc == 0 else "failed",
                                  exit_code=rc, seconds=time.time() - start)
        save()
        if rc:
            raise RuntimeError(f"{key} exited {rc}; inspect preserved log")
    finally:
        fcntl.flock(lease, fcntl.LOCK_UN)
        lease.close()
        AVAILABLE.put_nowait(gpu)


async def generate(sample, label):
    await execute(sample, "generate", G / "generate_dev6.py", label=label)
    await execute(sample, "validate", G / "validate_dev6_generation.py", (sample, label), label=label)


async def finish_pair(sample):
    results = await asyncio.gather(*(generate(sample, label) for label in ("G1", "G2")),
                                   return_exceptions=True)
    errors = [repr(x) for x in results if isinstance(x, BaseException)]
    if errors:
        raise RuntimeError(str(errors))


async def remaining(sample):
    await execute(sample, "prepare", G / "prepare_dev6_generation.py", (sample,))
    await finish_pair(sample)


async def main():
    assert not (B / "status.json").exists(), "Inspect previous state before recovery"
    assert sha(B / "PROTOCOL.json") == (B / "PROTOCOL.sha256").read_text().strip()
    query = await asyncio.create_subprocess_exec(
        "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader",
        stdout=asyncio.subprocess.PIPE)
    stdout, _ = await query.communicate()
    assert query.returncode == 0
    UUIDS.update({int(line.split(", ")[0]): line.split(", ")[1] for line in stdout.decode().splitlines()})
    assert set(UUIDS) == set(range(8))
    for gpu in range(8):
        AVAILABLE.put_nowait(gpu)
    save()
    # Pilot pair geometry must validate before scheduling any other subject.
    await execute("P18", "prepare", G / "prepare_dev6_generation.py", ("P18",))
    subjects = json.loads((B / "PROTOCOL.json").read_text())["subjects"]
    results = await asyncio.gather(finish_pair("P18"),
                                   *(remaining(s) for s in subjects if s != "P18"),
                                   return_exceptions=True)
    STATE["errors"] = [repr(x) for x in results if isinstance(x, BaseException)]
    STATE["state"] = "queue_failed_preserved_results" if STATE["errors"] else "generated_validated_needs_visual_review"
    save()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except BaseException as error:
        STATE.update(state="controller_failed", error=repr(error))
        save()
        raise
