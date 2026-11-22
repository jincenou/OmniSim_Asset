"""Eight asynchronous GPU-leased preparation workers, generation prohibited."""
import asyncio
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

from scan_native import ROOT, sha, write

W = ROOT.parent
ENV = Path("/mnt/ojc/miniconda3/envs")
LOCKS = W / "gpu_leases_20261019"


async def main():
    path = ROOT / "foreground_retry_prepare_status.json"
    if path.exists():
        raise RuntimeError("Explicit recovery required; never overwrite running state")
    manifest = ROOT / "foreground_retry_queue.json"
    rows = json.loads(manifest.read_text())["candidates"]
    frozen = {str(p): sha(p) for p in sorted((ROOT / "scripts_processed").glob("*.py"))}
    frozen[str(ROOT / "INPUT_GATE_PROTOCOL.json")] = sha(ROOT / "INPUT_GATE_PROTOCOL.json")
    frozen[str(manifest)] = sha(manifest)
    write(ROOT / "foreground_retry_prepare_code_frozen.json", frozen)
    state = {"state": "running", "started": time.time(), "target_candidates": len(rows),
             "jobs": {}, "workers": {}, "generation_started": False}
    queue = asyncio.Queue()
    for row in rows:
        queue.put_nowait(row)
    stages = [("camera", "4dgaussians", ["processed_camera.py"]),
              ("crop", "4danyone", ["m02_prepare.py"]),
              ("interpolate", "4danyone", ["interpolate.py"]),
              ("seg_prepare", "4danyone", ["source_foreground.py", "prepare"]),
              ("segment", "4danyone-sam2", ["source_foreground.py", "segment"]),
              ("composite", "4danyone", ["source_foreground.py", "composite"]),
              ("motion", "4danyone", ["motion.py"]),
              ("conditioning", "4danyone", ["conditioning_prepare_only.py"]),
              ("finish", "4danyone", ["finish_input.py"])]

    async def worker(gpu):
        while not queue.empty():
            state["workers"][str(gpu)] = "waiting_for_lease"
            write(path, state)
            lease = (LOCKS / f"gpu{gpu}.lock").open("a")
            try:
                await asyncio.to_thread(fcntl.flock, lease, fcntl.LOCK_EX)
                query = await asyncio.create_subprocess_exec(
                    "nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits", stdout=asyncio.subprocess.PIPE)
                result, _ = await query.communicate()
                if query.returncode or int(result.strip()) < 16000 or shutil.disk_usage(ROOT).free < 8*2**30:
                    state["workers"][str(gpu)] = "waiting_for_memory_or_disk"
                    write(path, state)
                    await asyncio.sleep(20)
                    continue
                try:
                    row = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                key, source = row["review_id"], Path(row["sample_root"])
                source.parent.mkdir(parents=True, exist_ok=False)
                write(source.parent / "candidate.json", row)
                runtime = W / "round2/moving61_runtime" if row["output_slots"] == 61 else W.parent / "repo"
                env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu),
                           OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                           PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                           FDANYONE_SAMPLE_ID=key, FDANYONE_SAMPLE_ROOT=str(source),
                           FDANYONE_MOVING_ROOT=str(source), FDANYONE_SOURCE_EXPERIMENT=str(source),
                           FDANYONE_WAYMO_ROOT=str(source.parent / "conditions"),
                           FDANYONE_MOTION_VARIANT="calibrated_moving", FDANYONE_ROOT_VIEWS="24",
                           FDANYONE_RUNTIME=str(runtime), FDANYONE_FRAMES=str(row["output_slots"]),
                           TORCH_EXTENSIONS_DIR=str(ROOT / "cache/torch_extensions"),
                           TORCH_CUDA_ARCH_LIST="8.9", MAX_JOBS="2", TMPDIR=str(ROOT / "tmp"))
                env.pop("PYTHONPATH", None)
                env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
                state["jobs"][key] = {"state": "running", "gpu": gpu, "source": str(source),
                                      "started": time.time(), "stages": {}}
                state["workers"][str(gpu)] = key
                write(path, state)
                for stage, environment, args in stages:
                    assert all(sha(Path(p)) == h for p, h in frozen.items()), "Frozen code changed"
                    if stage == "conditioning":
                        write(source / "research/motion_gate.json", {
                            "input_sha256": sha(source / "inputs/main.mp4"),
                            "kind": "technical_preparation_only", "visual_review": "pending",
                            "variants": {"calibrated_moving": {"allow_conditioning": True,
                                "motion_sha256": sha(source / "motion_pair/calibrated_moving/motion.safetensors")}},
                            "allow_generation": False})
                    command = [str(ENV / environment / "bin/python"), str(ROOT / "scripts_processed" / args[0]), *args[1:]]
                    if stage == "camera":
                        command += ["--review-id", key, "--manifest", str(manifest)]
                    with (ROOT / "logs" / f"{key}_{stage}.log").open("x") as log:
                        child = await asyncio.create_subprocess_exec(*command, env=env, stdout=log, stderr=log)
                        job = state["jobs"][key]
                        job.update(stage=stage, pid=child.pid)
                        job["stages"][stage] = {"started": time.time()}
                        write(path, state)
                        code = await child.wait()
                    job["stages"][stage].update(exit_code=code, finished=time.time())
                    write(path, state)
                    if code:
                        job.update(state="preparation_failed_not_identity_rejected",
                                   failed_stage=stage, finished=time.time())
                        write(path, state)
                        break
                else:
                    assert (source.parent / "READY.json").is_file()
                    job.update(state="prepared_review_pending", finished=time.time())
                    write(path, state)
                queue.task_done()
            finally:
                fcntl.flock(lease, fcntl.LOCK_UN)
                lease.close()
    write(path, state)
    await asyncio.gather(*(worker(g) for g in range(8)))
    state.update(state="finished_requires_review", finished=time.time())
    write(path, state)


if __name__ == "__main__":
    asyncio.run(main())
