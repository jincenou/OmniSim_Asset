"""Six fixed old/new second-seed pairs, sharing existing GPU leases."""
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from run_reference_anchor import gpu_lease

G = Path(__file__).resolve().parent
V = G / "validation12"
S = V / "second_seed"
PY = "/mnt/ojc/miniconda3/envs/4danyone/bin/python"
CPU = asyncio.Semaphore(2)
STATE = {"state": "running_or_waiting", "jobs": {}, "quality_approved": False}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save():
    STATE["updated_at"] = time.time()
    path = S / "status.writing"
    path.write_text(json.dumps(STATE, indent=2) + "\n")
    path.replace(S / "status.json")


def frozen(cfg):
    assert sha(S / "PROTOCOL.json") == (S / "PROTOCOL.sha256").read_text().strip()
    for filename, expected in cfg["code_sha256"].items():
        assert sha(Path(filename)) == expected, filename


async def execute(sample, label, stage, cfg):
    frozen(cfg)
    frames = cfg["frames"][sample]
    key = f"{sample}_{label}_{stage}"
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.update(CUDA_VISIBLE_DEVICES="", FDANYONE_FRAMES=str(frames),
               FDANYONE_RUNTIME=str(G.parent / "round2/moving61_runtime" if frames == 61 else G.parents[1] / "repo"),
               FDANYONE_WAYMO_ROOT=str(S / sample / label), FDANYONE_RETAINED_VIEWS="24",
               TORCH_EXTENSIONS_DIR=str(G.parent / "scene20/r20261019/cache/torch_extensions"),
               TORCH_CUDA_ARCH_LIST="8.9", MAX_JOBS="2", TMPDIR=str(S / "tmp"))
    lease = None
    STATE["jobs"][key] = {"state": "waiting_gpu" if stage == "generate" else "starting_cpu"}
    save()
    if stage == "generate":
        for filename, expected in read(S / sample / "conditioning_audit.json")["input_sha256"].items():
            assert sha(Path(filename)) == expected, filename
        gpu, uuid, lease = await gpu_lease(frames, cfg)
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   FD_EXPECTED_GPU_UUID=uuid)
        script, args = G / "generate_validation_seed.py", ()
    elif stage == "prepare":
        script, args = G / "prepare_validation_seed.py", (sample,)
    else:
        script, args = G / "validate_validation_seed.py", (sample, label)
    try:
        start = time.time()
        with (S / "logs" / f"{key}.log").open("x") as log:
            proc = await asyncio.create_subprocess_exec(PY, str(script), *args, env=env, stdout=log, stderr=log)
            STATE["jobs"][key] = {"state": "running", "pid": proc.pid, "started": start,
                                  "uuid": env.get("FD_EXPECTED_GPU_UUID")}
            save()
            code = await proc.wait()
        STATE["jobs"][key].update(state="completed" if code == 0 else "failed",
                                  exit_code=code, seconds=time.time() - start)
        save()
        assert code == 0, f"{key} exited {code}"
    finally:
        if lease is not None:
            fcntl.flock(lease, fcntl.LOCK_UN)
            lease.close()


async def pair(sample, cfg):
    # Do not bypass a failed primary candidate or consume duplicate generation budget.
    while not (V / "generation" / sample / "G2/validation.json").exists():
        state = read(V / "generation/status.json")
        if state["state"] in ("completed_with_failures", "controller_failed", "generated_validated_needs_quality_review"):
            raise RuntimeError(f"{sample}: no validated first-seed candidate")
        await asyncio.sleep(20)
    audit = S / sample / "conditioning_audit.json"
    if audit.exists():
        record = read(audit)
        assert record["state"] == "both_conditioning_sets_prepared"
        assert record["sample"] == sample and record["seed"] == cfg["seed"]
        for filename, expected in record["input_sha256"].items():
            assert sha(Path(filename)) == expected, filename
        STATE["jobs"][f"{sample}_pair_prepare"] = {"state": "verified_existing_preparation"}
        save()
    else:
        async with CPU:
            await execute(sample, "pair", "prepare", cfg)

    async def arm(label):
        await execute(sample, label, "generate", cfg)
        async with CPU:
            await execute(sample, label, "validate", cfg)

    outcomes = await asyncio.gather(arm("G0"), arm("G2"), return_exceptions=True)
    errors = [repr(x) for x in outcomes if isinstance(x, BaseException)]
    if errors:
        raise RuntimeError("; ".join(errors))


async def main():
    assert not (S / "status.json").exists()
    cfg = read(S / "PROTOCOL.json")
    assert len(cfg["subjects"]) == 6 and cfg["seed"] == 314159
    frozen(cfg)
    save()
    outcomes = await asyncio.gather(*(pair(s, cfg) for s in cfg["subjects"]), return_exceptions=True)
    errors = {s: repr(r) for s, r in zip(cfg["subjects"], outcomes) if isinstance(r, BaseException)}
    STATE.update(state="completed_with_failures" if errors else "generated_validated_quality_pending", errors=errors)
    save()


if __name__ == "__main__":
    asyncio.run(main())
