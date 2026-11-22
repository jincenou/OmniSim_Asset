"""Bounded async G3 queue; generation uses GPU leases, validation is CPU-only."""
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

G = Path(__file__).resolve().parent
B = G / "generation_dev6"
A = B / "reference_anchor"
PY = "/mnt/ojc/miniconda3/envs/4danyone/bin/python"
STATE = {"state": "waiting_or_running", "jobs": {}, "quality_approved": False}
MUTEX = asyncio.Lock()
CPU = asyncio.Semaphore(2)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save():
    STATE["updated_at"] = time.time()
    tmp = A / "status.writing"
    tmp.write_text(json.dumps(STATE, indent=2) + "\n")
    tmp.replace(A / "status.json")


def check_frozen(protocol):
    assert sha(A / "PROTOCOL.json") == (A / "PROTOCOL.sha256").read_text().strip()
    for file, digest in protocol["code_sha256"].items():
        assert sha(Path(file)) == digest, file


async def gpu_lease(frames, protocol):
    # Calibrated against eight completed 61-frame arms: peak reserved 25,238 MiB.
    minimum = protocol["minimum_free_mib"][str(frames)]
    while True:
        async with MUTEX:
            query = await asyncio.create_subprocess_exec(
                "nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE)
            raw, _ = await query.communicate()
            assert query.returncode == 0
            devices = [line.split(", ") for line in raw.decode().splitlines()]
            devices.sort(key=lambda d: int(d[2]), reverse=True)
            for physical, uuid, free in devices:
                gpu = int(physical)
                assert uuid == protocol["gpu_uuids"][physical]
                if int(free) < minimum or shutil.disk_usage(B).free < 8 * 2**30:
                    continue
                lease = (G.parent / "gpu_leases_20261019" / f"gpu{gpu}.lock").open("a")
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close()
                    continue
                return gpu, uuid, lease
        await asyncio.sleep(15)


async def job(sample, protocol):
    parent = B / sample / "G2"
    root = B / sample / "G3"
    STATE["jobs"][sample] = dict(state="waiting_for_G2_validation")
    save()
    while not (parent / "validation.json").exists():
        status_path = parent / "status.json"
        if status_path.exists() and json.loads(status_path.read_text()).get("state") == "failed":
            raise RuntimeError(f"{sample} parent failed; no G3 start")
        await asyncio.sleep(15)
    assert json.loads((parent / "validation.json").read_text())["state"] == "complete_decoding_and_stage_evidence_verified"
    for path, digest in protocol["conditions_sha256"][sample].items():
        assert sha(Path(path)) == digest, path
    check_frozen(protocol)
    assert not root.exists()
    root.mkdir()
    (root / "inputs").symlink_to(parent / "inputs", target_is_directory=True)
    (root / "preflight").symlink_to(parent / "preflight", target_is_directory=True)
    (root / "condition_reuse.json").write_text(json.dumps(dict(
        source=str(parent), exact_condition_hashes=protocol["conditions_sha256"][sample],
        policy="Identical G2 pose, camera, crop, source and four-reference packing; only target anchor intervention"),
        indent=2) + "\n")
    count = protocol["frames"][sample]
    STATE["jobs"][sample] = dict(state="waiting_gpu", frames=count)
    save()
    gpu, uuid, lease = await gpu_lease(count, protocol)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
               FD_EXPECTED_GPU_UUID=uuid, FDANYONE_FRAMES=str(count),
               FDANYONE_RUNTIME=str(G.parent / "round2/moving61_runtime" if count == 61 else G.parent.parent / "repo"),
               FDANYONE_WAYMO_ROOT=str(root), FDANYONE_RETAINED_VIEWS="24",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TORCH_EXTENSIONS_DIR=str(G.parent / "scene20/r20261019/cache/torch_extensions"),
               TORCH_CUDA_ARCH_LIST="8.9", MAX_JOBS="2", TMPDIR=str(B / "tmp"))
    try:
        check_frozen(protocol)
        start = time.time()
        with (A / "logs" / f"{sample}_generate.log").open("x") as log:
            proc = await asyncio.create_subprocess_exec(
                PY, str(G / "generate_reference_anchor.py"), env=env, stdout=log, stderr=log)
            STATE["jobs"][sample] = dict(state="generating", gpu=gpu, uuid=uuid, pid=proc.pid, started=start)
            save()
            rc = await proc.wait()
        STATE["jobs"][sample].update(generation_exit_code=rc, generation_seconds=time.time() - start)
        if rc:
            STATE["jobs"][sample]["state"] = "failed_generation"
            save()
            raise RuntimeError(f"{sample} generation exited {rc}")
    finally:
        fcntl.flock(lease, fcntl.LOCK_UN)
        lease.close()
    STATE["jobs"][sample]["state"] = "waiting_cpu_validation"
    save()
    async with CPU:
        env["CUDA_VISIBLE_DEVICES"] = ""
        with (A / "logs" / f"{sample}_validate.log").open("x") as log:
            proc = await asyncio.create_subprocess_exec(
                PY, str(G / "validate_reference_anchor.py"), sample, env=env, stdout=log, stderr=log)
            STATE["jobs"][sample].update(state="validating_cpu", validation_pid=proc.pid)
            save()
            rc = await proc.wait()
        STATE["jobs"][sample].update(validation_exit_code=rc,
            state="validated_needs_quality_review" if rc == 0 else "failed_validation")
        save()
        if rc:
            raise RuntimeError(f"{sample} validation exited {rc}")


async def main():
    assert not (A / "status.json").exists(), "Inspect existing controller before recovery"
    protocol = json.loads((A / "PROTOCOL.json").read_text())
    check_frozen(protocol)
    save()
    results = await asyncio.gather(*(job(s, protocol) for s in protocol["subjects"]), return_exceptions=True)
    STATE["errors"] = [repr(r) for r in results if isinstance(r, BaseException)]
    STATE["state"] = "failed_preserved_results" if STATE["errors"] else "complete_needs_quality_review"
    save()


if __name__ == "__main__":
    asyncio.run(main())
