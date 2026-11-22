"""Prepare six paired Grid4D inputs; this queue does not launch reconstruction."""
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
D = V / "downstream"
ENV = Path("/mnt/ojc/miniconda3/envs")
MATERIALIZE = asyncio.Semaphore(2)
STATE = {"state": "preparing_or_waiting", "jobs": {}, "quality_approved": False}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save():
    STATE["updated_at"] = time.time()
    p = D / "PREP_STATUS.writing"
    p.write_text(json.dumps(STATE, indent=2) + "\n")
    p.replace(D / "PREP_STATUS.json")


def frozen(cfg):
    assert sha(D / "PREP_PROTOCOL.json") == (D / "PREP_PROTOCOL.sha256").read_text().strip()
    for name, expected in cfg["code_sha256"].items():
        assert sha(Path(name)) == expected, name


async def execute(sample, arm, stage, cfg):
    frozen(cfg)
    label = sample + "_" + arm
    key = label + "_" + stage
    frames = cfg["frames"][sample]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               FDANYONE_ASSET_NAME=label, FDANYONE_ASSET_SOURCE=str(D / "sources" / label),
               FDANYONE_DATA=str(D / "data" / label), FDANYONE_FRAMES=str(frames),
               TMPDIR=str(D / "tmp"))
    env.pop("PYTHONPATH", None)
    lease = None
    STATE["jobs"][key] = {"state": "waiting_gpu" if stage == "masks" else "starting_cpu"}
    save()
    if stage == "source":
        python, script, args = ENV / "4danyone/bin/python", G / "prepare_downstream_source.py", (sample, arm)
    elif stage == "masks":
        gpu, uuid, lease = await gpu_lease(frames, cfg)
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   FDANYONE_ASSET_GPU_IDS=str(gpu), FD_EXPECTED_GPU_UUID=uuid)
        python, script, args = ENV / "4danyone-sam2/bin/python", G / "downstream_masks.py", ()
    elif stage == "hull":
        python, script, args = ENV / "4danyone-ftgs/bin/python", D / "scripts/export_sequence.py", ()
    else:
        python, script, args = ENV / "4danyone-ftgs/bin/python", D / "scripts/export_4dgs.py", (
            "--label", label, "--data", str(D / "data" / label))
    try:
        start = time.time()
        with (D / "logs" / f"{key}.log").open("x") as log:
            proc = await asyncio.create_subprocess_exec(str(python), str(script), *args, env=env, stdout=log, stderr=log)
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


async def case(sample, arm, cfg):
    while not (V / "generation" / sample / "G2/validation.json").exists():
        state = read(V / "generation/status.json")
        if state["state"] in ("completed_with_failures", "controller_failed", "generated_validated_needs_quality_review"):
            raise RuntimeError(f"{sample}: missing complete candidate; pair preparation stopped")
        await asyncio.sleep(20)
    async with MATERIALIZE:
        await execute(sample, arm, "source", cfg)
        await execute(sample, arm, "masks", cfg)
        await execute(sample, arm, "hull", cfg)
        await execute(sample, arm, "grid_manifest", cfg)


async def main():
    assert not (D / "PREP_STATUS.json").exists()
    cfg = read(D / "PREP_PROTOCOL.json")
    assert len(cfg["subjects"]) == 6
    frozen(cfg)
    save()
    labels = [(s, a) for s in cfg["subjects"] for a in ("G0", "G2")]
    results = await asyncio.gather(*(case(s, a, cfg) for s, a in labels), return_exceptions=True)
    errors = {f"{s}_{a}": repr(r) for (s, a), r in zip(labels, results) if isinstance(r, BaseException)}
    STATE.update(state="prepared_with_failures" if errors else "all_inputs_prepared_not_trained",
                 errors=errors)
    save()


if __name__ == "__main__":
    asyncio.run(main())
