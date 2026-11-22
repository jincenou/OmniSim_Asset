"""Fixed candidate queue: verify old baseline, prepare, generate and decode."""
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
B = V / "generation"
PY = "/mnt/ojc/miniconda3/envs/4danyone/bin/python"
STATE = dict(state="starting", jobs={}, quality_approved=False)
CPU = asyncio.Semaphore(2)
PREP = asyncio.Semaphore(2)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save():
    STATE["updated_at"] = time.time()
    p = B / "status.writing"
    p.write_text(json.dumps(STATE, indent=2) + "\n")
    p.replace(B / "status.json")


async def execute(sample, stage, cfg, script, args=(), gpu_stage=False):
    for filename, digest in cfg["code_sha256"].items():
        assert sha(Path(filename)) == digest, filename
    key = sample + "_" + stage
    STATE["jobs"][key] = dict(state="waiting_gpu" if gpu_stage else "starting_cpu")
    save()
    count = cfg["frames"][sample]
    lease = None
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    if gpu_stage:
        resource_cfg = dict(cfg)
        if stage == "prepare":
            resource_cfg["minimum_free_mib"] = {"61": 12000, "121": 14000}
        gpu, uuid, lease = await gpu_lease(count, resource_cfg)
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   FD_EXPECTED_GPU_UUID=uuid)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    env.update(
        FDANYONE_FRAMES=str(count),
        FDANYONE_RUNTIME=str(G.parent / "round2/moving61_runtime" if count == 61 else G.parents[1] / "repo"),
        FDANYONE_WAYMO_ROOT=str(B / sample / "G2"),
        FDANYONE_RETAINED_VIEWS="24",
        TORCH_EXTENSIONS_DIR=str(G.parent / "scene20/r20261019/cache/torch_extensions"),
        TORCH_CUDA_ARCH_LIST="8.9", MAX_JOBS="2", TMPDIR=str(B / "tmp"))
    try:
        with (B / "logs" / (key + ".log")).open("x") as log:
            started = time.time()
            child = await asyncio.create_subprocess_exec(PY, str(script), *args, env=env, stdout=log, stderr=log)
            STATE["jobs"][key] = dict(state="running", pid=child.pid, started=started,
                                     uuid=env.get("FD_EXPECTED_GPU_UUID"))
            save()
            rc = await child.wait()
        STATE["jobs"][key].update(state="completed" if rc == 0 else "failed",
                                  exit_code=rc, seconds=time.time() - started)
        save()
        if rc:
            raise RuntimeError(f"{key} exited {rc}")
    finally:
        if lease is not None:
            fcntl.flock(lease, fcntl.LOCK_UN)
            lease.close()


async def prepare(sample, cfg):
    while True:
        motion = json.loads((V / "motion/status.json").read_text())
        job = motion["jobs"].get(sample + "_align", {})
        if job.get("state") == "completed":
            assert job["exit_code"] == 0
            break
        if job.get("state") == "failed" or sample in motion.get("errors", {}):
            raise RuntimeError(f"{sample} motion failed; retain in denominator")
        await asyncio.sleep(15)
    aligned = V / "motion/cases" / sample / "text/aligned/motion_candidates.safetensors"
    report = aligned.with_name("report.json")
    assert json.loads(report.read_text())["protocol_sha256"] == sha(V / "MOTION_ALIGN_PROTOCOL.json")
    receipt = B / "receipts" / (sample + ".json")
    assert not receipt.exists()
    receipt.write_text(json.dumps(dict(
        aligned_sha256=sha(aligned), alignment_report_sha256=sha(report),
        motion_recipe_sha256=sha(V / "MOTION_PROTOCOL.json"),
        state="completed_candidate_not_quality_approved"), indent=2) + "\n")
    async with CPU:
        await execute(sample, "old_check", cfg, G / "validate_validation_old.py", (sample,))
    async with PREP:
        await execute(sample, "prepare", cfg, G / "prepare_validation_generation.py", (sample,), gpu_stage=True)


async def finish(sample, cfg):
    await execute(sample, "generate", cfg, G / "generate_dev6.py", gpu_stage=True)
    async with CPU:
        await execute(sample, "validate", cfg, G / "validate_validation_new.py", (sample, "G2"))


async def subject(sample, cfg):
    await prepare(sample, cfg)
    await finish(sample, cfg)


async def main():
    assert not (B / "status.json").exists()
    path = B / "PROTOCOL.json"
    assert sha(path) == path.with_suffix(".sha256").read_text().strip()
    cfg = json.loads(path.read_text())
    for name in ("logs", "receipts", "old_checks", "tmp"):
        (B / name).mkdir(exist_ok=False)
    save()
    try:
        # Fixed first subject, selected before any validation output exists.
        first, *others = cfg["subjects"]
        await prepare(first, cfg)
        STATE["state"] = "generation_and_preparation_running"
        save()
        outcomes = await asyncio.gather(finish(first, cfg), *(subject(s, cfg) for s in others), return_exceptions=True)
        errors = {s: repr(result) for s, result in zip(cfg["subjects"], outcomes) if isinstance(result, BaseException)}
        STATE.update(state="completed_with_failures" if errors else "generated_validated_needs_quality_review",
                     errors=errors)
    except Exception as exc:
        STATE.update(state="controller_failed", error=repr(exc))
    save()


if __name__ == "__main__":
    asyncio.run(main())
