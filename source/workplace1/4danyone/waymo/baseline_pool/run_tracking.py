"""Async GPU-leased baseline tracking, independently resumable per scene."""
import argparse
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parent
PY = "/mnt/ojc/miniconda3/envs/4danyone/bin/python"
LOCKS = ROOT.parent / "gpu_leases_20261019"


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    suffix = "pilot" if args.pilot else "tracking"
    path = ROOT / (suffix + "_status.json")
    state = {"state": "running", "jobs": {}, "generation_started": False}
    def save():
        state["updated_at"] = time.time()
        tmp = path.with_suffix(".writing")
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(path)
    if path.exists():
        previous = json.loads(path.read_text())
        assert previous["state"] in ("completed", "completed_with_failures")
        state["jobs"] = previous["jobs"]
    rows = json.loads((ROOT / "inventory.json").read_text())["scenes"]
    if args.pilot:
        rows = [r for r in rows if r["split"] == "train" and r["scene"] in ("001", "010")]
    else:
        gate = json.loads((ROOT / "tracking_admission.json").read_text())
        assert gate["allow_full_tracking"] is True
        for name, expected in gate["code_sha256"].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
        rows = [r for r in rows if r["native_labels_tfrecord"] is None]
    queue = asyncio.Queue()
    for row in rows:
        label = row["split"] + "_" + row["scene"]
        receipt = ROOT / ("pilot" if args.pilot else "predicted_tracks") / row["split"] / row["scene"] / "receipt.json"
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            assert saved["source_metadata_sha256"] == hashlib.sha256(Path(row["metadata"]).read_bytes()).hexdigest()
            assert saved["worker_sha256"] == hashlib.sha256((ROOT / "track_scene.py").read_bytes()).hexdigest()
            assert saved["protocol_sha256"] == hashlib.sha256((ROOT / "SCREEN_PROTOCOL.json").read_bytes()).hexdigest()
            assert saved["tracker_sha256"] == hashlib.sha256((ROOT / "bytetrack.yaml").read_bytes()).hexdigest()
            assert len(saved["cameras"]) == len(row["camera_indices"])
            assert all(c["frame_count"] == row["frames"] for c in saved["cameras"])
            state["jobs"][label] = {"state": "completed", "reuse": str(receipt), "exit_code": 0}
            continue
        if state["jobs"].get(label, {}).get("state") == "completed":
            raise RuntimeError(f"Completed state without receipt: {label}")
        queue.put_nowait(row)
    async def worker(gpu):
        while not queue.empty():
            lease = (LOCKS / f"gpu{gpu}.lock").open("a")
            try:
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    await asyncio.sleep(15)
                    continue
                proc = await asyncio.create_subprocess_exec("nvidia-smi", f"--id={gpu}",
                    "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits",
                    stdout=asyncio.subprocess.PIPE)
                raw, _ = await proc.communicate()
                assert proc.returncode == 0
                uuid, free = [x.strip() for x in raw.decode().strip().split(",")]
                if int(free) < 8192 or shutil.disk_usage(ROOT).free < 8 * 2**30:
                    await asyncio.sleep(15)
                    continue
                try:
                    row = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                label = row["split"] + "_" + row["scene"]
                logpath = ROOT / "logs" / f"{suffix}_{label}.log"
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, FD_EXPECTED_GPU_UUID=uuid,
                    CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                    OPENBLAS_NUM_THREADS="2", PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1",
                    PYTHONDONTWRITEBYTECODE="1", YOLO_AUTOINSTALL="false",
                    YOLO_CONFIG_DIR=str(ROOT / "cache/ultralytics"),
                    MPLCONFIGDIR=str(ROOT / "cache/matplotlib"))
                command = [PY, str(ROOT / "track_scene.py"), "--split", row["split"], "--scene", row["scene"]]
                if args.pilot:
                    command += ["--pilot"]
                with logpath.open("x") as log:
                    child = await asyncio.create_subprocess_exec(*command, env=env, stdout=log, stderr=log)
                    state["jobs"][label] = {"state": "running", "pid": child.pid, "gpu": gpu,
                                          "uuid": uuid, "started": time.time()}
                    save()
                    code = await child.wait()
                state["jobs"][label].update(state="completed" if code == 0 else "failed", exit_code=code)
                save()
                queue.task_done()
            finally:
                fcntl.flock(lease, fcntl.LOCK_UN)
                lease.close()
    save()
    await asyncio.gather(*(worker(gpu) for gpu in range(8)))
    state["state"] = "completed_with_failures" if any(
        v["state"] != "completed" for v in state["jobs"].values()) else "completed"
    save()


if __name__ == "__main__":
    asyncio.run(main())
