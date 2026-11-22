"""Low-frequency telemetry for this task only; no credentials or request URLs."""
import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


async def capture(*args):
    process = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    data, _ = await process.communicate()
    return process.returncode, data.decode()


async def sample():
    process_result, sockets_result, gpu_result = await asyncio.gather(
        capture("pgrep", "-f", r"^/mnt/ojc/miniconda3/envs/4danyone/bin/python /mnt/ojc/workplace1/4danyone/scripts/download_http.py"),
        capture("ss", "-tinp"),
        capture("nvidia-smi", "--id=4,5,6,7", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"),
    )
    pids = set(process_result[1].split())
    lines = sockets_result[1].splitlines()
    sockets = {}
    for index, line in enumerate(lines[:-1]):
        match = re.search(r"pid=(\d+),", line)
        if match and match.group(1) in pids:
            received = re.search(r"bytes_received:(\d+)", lines[index + 1])
            if received:
                key = match.group(1) + ":" + " ".join(line.split()[:5])
                sockets[key] = int(received.group(1))
    return pids, sockets, gpu_result


async def main():
    previous = {}
    last = time.monotonic()
    while True:
        pids, sockets, gpu = await sample()
        now = time.monotonic()
        delta = sum(map(lambda pair: max(0, pair[1] - previous.get(pair[0], pair[1])), sockets.items()))
        telemetry = {
            "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "download_processes": sorted(pids),
            "active_download_connections": len(sockets),
            "observed_tcp_receive_MiB_per_second": delta / max(now - last, 1) / 2**20,
            "rate_note": "Lower-bound interval rate for surviving task sockets; excludes closed connections and first sample.",
            "gpu_query_exit_code": gpu[0],
            "gpu_csv": gpu[1].strip(),
        }
        temporary = ROOT / "logs/telemetry.json.tmp"
        temporary.write_text(json.dumps(telemetry, indent=2) + "\n")
        temporary.replace(ROOT / "logs/telemetry.json")
        previous, last = sockets, now
        state = json.loads((ROOT / "status.json").read_text())
        if state["state"] in {"failed", "validated"}:
            return
        await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
