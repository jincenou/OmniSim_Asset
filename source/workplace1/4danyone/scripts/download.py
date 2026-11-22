"""Fetch pinned public assets with bounded asynchronous concurrency."""
import asyncio
import hashlib
import json
import shutil
import time
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "repo"
LIMIT = asyncio.Semaphore(3)
RESERVE = 10 * 1024**3


def entries():
    model = json.loads((ROOT / "research/model.json").read_text())
    foreground = json.loads((ROOT / "research/birefnet.json").read_text())
    def main_asset(item):
        name = item["rfilename"]
        return name.startswith(("4danyone/", "gvhmr/", "perceptual/", "licenses/")) or name in {
            "LICENSE.md", "README.md", "data/source/pexels/2785536-uhd_2160_3840_25fps.mp4"
        }
    def pack(meta, prefix, item):
        name = item["rfilename"]
        dest = REPO / name if name.startswith("data/") else REPO / "models" / prefix / name
        return {"url": f"https://huggingface.co/{meta['id']}/resolve/{meta['sha']}/{quote(name)}", "path": str(dest), "size": item["size"], "sha256": item.get("lfs", {}).get("sha256")}
    first = map(lambda item: pack(model, "", item), filter(main_asset, model["siblings"]))
    second = map(lambda item: pack(foreground, "birefnet", item), filter(lambda item: item["rfilename"] in {"BiRefNet_config.py", "birefnet.py", "config.json", "model.safetensors"}, foreground["siblings"]))
    return [*first, *second]


async def fetch(item):
    async with LIMIT:
        path = Path(item["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        if not path.exists():
            process = await asyncio.create_subprocess_exec(
                "/mnt/ojc/miniconda3/envs/4danyone/bin/python",
                str(ROOT / "scripts/hf_fetch.py"), json.dumps(item)
            )
            try:
                if await process.wait():
                    raise RuntimeError(f"Download failed: {path.name}")
            finally:
                if process.returncode is None:
                    process.terminate()
                    await process.wait()
        if path.stat().st_size != item["size"]:
            raise RuntimeError(f"Size mismatch: {path.name}")
        if item["sha256"]:
            process = await asyncio.create_subprocess_exec("sha256sum", str(path), stdout=asyncio.subprocess.PIPE)
            output, _ = await process.communicate()
            if process.returncode or output.decode().split()[0] != item["sha256"]:
                raise RuntimeError(f"SHA256 mismatch: {path.name}")
        partial.unlink(missing_ok=True)
        print(json.dumps({"verified": str(path.relative_to(ROOT)), "bytes": item["size"]}), flush=True)


def byte_count(items):
    def size(item):
        path = Path(item["path"])
        partial = path.with_name(path.name + ".part")
        return path.stat().st_size if path.exists() else 0
    return sum(map(size, items))


async def monitor(items):
    previous = byte_count(items)
    timestamp = time.monotonic()
    while True:
        await asyncio.sleep(10)
        free = shutil.disk_usage(ROOT).free
        if free < RESERVE:
            raise RuntimeError("Free disk space below 10 GiB; stopping this download task")
        elapsed = time.monotonic() - timestamp
        if elapsed >= 300:
            current = byte_count(items)
            staged = sum(map(lambda p: p.stat().st_blocks * 512, REPO.rglob("*.incomplete")))
            state = {"transport": "huggingface_hub_xet", "completed_bytes": current, "total_bytes": sum(map(lambda item: item["size"], items)), "completed_MiB_per_second": (current - previous) / elapsed / 2**20, "incomplete_disk_bytes": staged, "network_speed": "not directly measured by Xet", "free_GiB": free / 2**30}
            (ROOT / "logs/download-status.json").write_text(json.dumps(state, indent=2) + "\n")
            print(json.dumps(state), flush=True)
            previous, timestamp = current, time.monotonic()


async def main():
    items = entries()
    (ROOT / "research/download-manifest.json").write_text(json.dumps(items, indent=2) + "\n")
    remaining = sum(map(lambda item: item["size"], items)) - byte_count(items)
    if shutil.disk_usage(ROOT).free < remaining + RESERVE:
        raise RuntimeError("Insufficient space for remaining assets plus 10 GiB reserve")
    print(json.dumps({"state": "started", "remaining_GiB": remaining / 2**30}), flush=True)
    async def all_downloads():
        await asyncio.gather(*map(fetch, items))
    job = asyncio.create_task(all_downloads())
    watcher = asyncio.create_task(monitor(items))
    try:
        done, _ = await asyncio.wait((job, watcher), return_when=asyncio.FIRST_COMPLETED)
        if watcher in done:
            await watcher
        await job
        (ROOT / "logs/download.complete").write_text("All pinned assets verified\n")
    finally:
        job.cancel()
        watcher.cancel()
        await asyncio.gather(job, watcher, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
