"""Frozen six-pair full-A reconstruction and isolated final asset replay."""
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from run_reference_anchor import gpu_lease

G = Path(__file__).resolve().parent
D = G / "validation12/downstream"
GRID = G.parent / "grid4d"
PY = "/mnt/ojc/miniconda3/envs/grid4d/bin/python"
CPU = asyncio.Semaphore(2)
STATE = {"state": "waiting_inputs_or_training", "jobs": {}, "quality_approved": False}


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save():
    STATE["updated_at"] = time.time()
    tmp = D / "TRAIN_STATUS.writing"
    tmp.write_text(json.dumps(STATE, indent=2) + "\n")
    tmp.replace(D / "TRAIN_STATUS.json")


def frozen(cfg):
    assert sha(D / "PLAN.json") == (D / "plan_hash.txt").read_text().strip()
    for name, expected in cfg["code_sha256"].items():
        assert sha(Path(name)) == expected, name


def runtime_environment():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CUDA_DEVICE_ORDER="PCI_BUS_ID",
               PATH="/mnt/ojc/miniconda3/envs/grid4d/bin:/mnt/ojc/miniconda3/envs/4dgaussians/bin:" + os.environ.get("PATH", ""),
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               CUDA_HOME="/mnt/ojc/miniconda3/envs/4dgaussians", CC="/usr/bin/gcc-9", CXX="/usr/bin/g++-9",
               TORCH_CUDA_ARCH_LIST="8.9", MAX_JOBS="2",
               TORCH_EXTENSIONS_DIR=str(GRID / "cache/extensions"),
               GRID4D_HASH_BUILD=str(GRID / "cache/extensions/hashencoder"),
               TORCH_HOME=str(GRID / "cache/torch"), TMPDIR=str(D / "tmp"),
               MPLCONFIGDIR=str(GRID / "cache/matplotlib"), HF_HUB_OFFLINE="1",
               CUDA_CACHE_PATH=str(GRID / "cache/cuda"), NUMBA_CACHE_DIR=str(GRID / "cache/numba"))
    env.pop("PYTHONPATH", None)
    assert shutil.which("ninja", path=env["PATH"]) is not None
    return env


async def execute(label, stage, cfg):
    frozen(cfg)
    sample = label.split("_")[0]
    frames = cfg["frames"][sample]
    env = runtime_environment()
    key = label + "_" + stage
    gpu_stage = stage not in ("audit_geometry", "configure")
    STATE["jobs"][key] = {"state": "waiting_gpu" if gpu_stage else "starting_cpu"}
    save()
    lease = None
    if gpu_stage:
        resource = dict(cfg)
        if stage == "replay":
            resource["minimum_free_mib"] = {"61": 6000, "121": 6000}
        _, uuid, lease = await gpu_lease(frames, resource)
        # The existing Grid4D runtime verifies a single physical UUID directly.
        env["CUDA_VISIBLE_DEVICES"] = uuid
    if stage == "audit_geometry":
        script, args = D / "scripts/audit_geometry.py", (label,)
    elif stage == "configure":
        script, args = D / "scripts/configure_grid.py", (label,)
    elif stage == "native":
        script, args = D / "scripts/train_prefix.py", (
            "--config", str(D / "configs" / f"{label}_33k.json"),
            "--run", str(D / "runs" / f"{label}_33k"))
    elif stage in ("alpha", "A"):
        script, args = D / "scripts/train_continuation.py", ("--label", label, "--stage", stage)
    else:
        script, args = D / "scripts/replay_final.py", (label,)
    try:
        start = time.time()
        with (D / "logs" / f"{key}.log").open("x") as log:
            proc = await asyncio.create_subprocess_exec(PY, str(script), *args, env=env, stdout=log, stderr=log)
            STATE["jobs"][key] = {"state": "running", "pid": proc.pid, "started": start,
                                  "uuid": env["CUDA_VISIBLE_DEVICES"] or None}
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


async def job(sample, arm, cfg):
    label = f"{sample}_{arm}"
    manifest = D / "four_dg/data" / label / "fdanyone.json"
    mask_receipt = D / "research" / f"{label}-mask-ready-complete.json"
    while not (manifest.exists() and mask_receipt.exists()):
        prep = read(D / "PREP_READY_STATUS.json")
        if prep["state"] in ("prepared_with_failures", "all_inputs_prepared_not_trained"):
            raise RuntimeError(f"{label}: preparation terminal without data")
        failed = {k: v for k, v in prep["jobs"].items() if k.startswith("ready_" + label + "_") and v["state"] == "failed"}
        if failed:
            raise RuntimeError(f"{label}: input stage failed")
        await asyncio.sleep(20)
    assert read(mask_receipt)["state"] == "24_view_RGBA_complete_not_quality_approved"
    assert read(mask_receipt)["views"] == 24
    async with CPU:
        await execute(label, "audit_geometry", cfg)
        await execute(label, "configure", cfg)
    await execute(label, "native", cfg)
    await execute(label, "alpha", cfg)
    await execute(label, "A", cfg)
    run = D / "runs" / f"{label}_A"
    source = D / "sources" / label
    shutil.copy2(source / "root_motion.npz", run / "root_motion.npz")
    receipt = read(source / "receipt.json")
    (run / "frontend_provenance.json").write_text(json.dumps({
        "sample": sample, "arm": arm, "generation_seed": 42,
        "root_motion_sha256": sha(run / "root_motion.npz"),
        "input_receipt": receipt, "quality_approved": False,
        "coordinates": "Asset uses source_to_normalized in asset_camera_data.json; root maps are in the pre-normalization frontend frame.",
        "scope": "Same frozen full-A recipe; not arbitrary action driving or metric Waymo-world registration.",
    }, indent=2) + "\n")
    await execute(label, "replay", cfg)


async def main():
    assert not (D / "TRAIN_STATUS.json").exists()
    cfg = read(D / "PLAN.json")
    frozen(cfg)
    save()
    labels = [(s, a) for s in cfg["subjects"] for a in ("G0", "G2")]
    assert len(labels) == 12
    results = await asyncio.gather(*(job(s, a, cfg) for s, a in labels), return_exceptions=True)
    errors = {f"{s}_{a}": repr(r) for (s, a), r in zip(labels, results) if isinstance(r, BaseException)}
    STATE.update(state="completed_with_failures" if errors else "all_A_assets_replayed_quality_pending", errors=errors)
    save()


if __name__ == "__main__":
    asyncio.run(main())
