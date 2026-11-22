"""Download pinned official assets with resumable, bounded async HTTP ranges."""
import asyncio
from collections import deque
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo

import httpx
from download import entries, ROOT

CHUNK = 32 * 1024**2
RESERVE = 10 * 1024**3
LIMIT = asyncio.Semaphore(6)
CACHE = ROOT / "cache/http_chunks"
STATS = {"received_bytes_this_run": 0, "verified_bytes": 0, "active_ranges": 0}


class InvalidRange(RuntimeError):
    pass


def verify(path, item):
    if path.stat().st_size != item["size"]:
        raise ValueError(f"Size mismatch: {path.name}")
    if item["sha256"]:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"SHA256 mismatch: {path.name}")


async def piece(client, item, folder, start):
    end = min(start + CHUNK, item["size"]) - 1
    length = end - start + 1
    target = folder / f"{start:016d}.chunk"
    if target.exists() and target.stat().st_size == length:
        return target
    temporary = target.with_suffix(".partial")
    attempt = 0
    while True:
        attempt += 1
        try:
            async with LIMIT:
                STATS["active_ranges"] += 1
                try:
                    async with client.stream(
                        "GET", item["url"],
                        params={"download": "true", "range_start": str(start), "nonce": str(time.time_ns())},
                        headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                    ) as response:
                        response.raise_for_status()
                        expected = f"bytes {start}-{end}/{item['size']}"
                        whole_file = response.status_code == 200 and start == 0 and end == item["size"] - 1
                        if not whole_file and (response.status_code != 206 or response.headers.get("content-range") != expected):
                            raise InvalidRange(f"Expected HTTP 206 and {expected}")
                        count = 0
                        with temporary.open("wb") as stream:
                            async for data in response.aiter_bytes(1024**2):
                                count += len(data)
                                if count > length:
                                    raise InvalidRange("Response exceeded requested length")
                                stream.write(data)
                                STATS["received_bytes_this_run"] += len(data)
                        if count != length:
                            raise InvalidRange("Incomplete range body")
                        temporary.replace(target)
                        return target
                finally:
                    STATS["active_ranges"] -= 1
        except (httpx.TransportError, httpx.HTTPStatusError, InvalidRange) as error:
            code = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
            retryable = code is None or code in {403, 408, 429, 500, 502, 503, 504}
            if not retryable or attempt >= 12:
                raise RuntimeError(f"Range failed: {Path(item['path']).name}, offset {start}, {type(error).__name__}, HTTP {code}, attempt {attempt}") from None
            delay = min(15 * 2 ** (attempt - 1), 300)
            print(json.dumps({"event": "range_retry", "file": Path(item["path"]).name, "offset": start, "attempt": attempt, "error_type": type(error).__name__, "http_status": code, "delay_seconds": delay}), flush=True)
            await asyncio.sleep(delay)


def assemble(paths, target, item):
    temporary = target.with_name(target.name + ".assembling")
    def append(path):
        with path.open("rb") as source:
            shutil.copyfileobj(source, output, length=4 * 1024**2)
    with temporary.open("wb") as output:
        deque(map(append, paths), maxlen=0)
    verify(temporary, item)
    temporary.replace(target)


async def asset(client, item):
    target = Path(item["path"])
    if target.exists():
        await asyncio.to_thread(verify, target, item)
    else:
        folder = CACHE / hashlib.sha256(item["url"].encode()).hexdigest()
        folder.mkdir(parents=True, exist_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        paths = await asyncio.gather(*map(lambda start: piece(client, item, folder, start), range(0, item["size"], CHUNK)))
        await asyncio.to_thread(assemble, paths, target, item)
        await asyncio.to_thread(shutil.rmtree, folder)
    STATS["verified_bytes"] += item["size"]
    print(json.dumps({"verified": str(target.relative_to(ROOT)), "bytes": item["size"]}), flush=True)


async def monitor(items):
    previous, last = 0, time.monotonic()
    while True:
        await asyncio.sleep(10)
        free = shutil.disk_usage(ROOT).free
        if free < RESERVE:
            raise RuntimeError("Free disk space below 10 GiB")
        now = time.monotonic()
        if now - last >= 300:
            state = {"updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "transport": "official_http_ranges", **STATS, "total_bytes": sum(map(lambda x: x["size"], items)), "received_MiB_per_second": (STATS["received_bytes_this_run"] - previous) / (now - last) / 2**20, "free_GiB": free / 2**30}
            temporary = ROOT / "logs/download-status.json.tmp"
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(ROOT / "logs/download-status.json")
            print(json.dumps(state), flush=True)
            previous, last = STATS["received_bytes_this_run"], now


async def main():
    items = entries()
    remaining = sum(map(lambda x: x["size"] if not Path(x["path"]).exists() else 0, items))
    if shutil.disk_usage(ROOT).free < 2 * remaining + RESERVE:
        raise RuntimeError("Need space for resumable chunks, assembled files, and 10 GiB reserve")
    CACHE.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "started", "transport": "official_http_ranges", "concurrency": 6, "chunk_MiB": 32, "remaining_GiB": remaining / 2**30}), flush=True)
    async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(90, connect=30), limits=httpx.Limits(max_connections=12)) as client:
        async def all_assets():
            await asyncio.gather(*map(lambda item: asset(client, item), items))
        job = asyncio.create_task(all_assets())
        watcher = asyncio.create_task(monitor(items))
        try:
            done, _ = await asyncio.wait((job, watcher), return_when=asyncio.FIRST_COMPLETED)
            if watcher in done:
                await watcher
            await job
            (ROOT / "logs/download.complete").write_text("All pinned assets verified\n")
            print(json.dumps({"event": "complete", "verified_bytes": STATS["verified_bytes"]}), flush=True)
        finally:
            job.cancel()
            watcher.cancel()
            await asyncio.gather(job, watcher, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
