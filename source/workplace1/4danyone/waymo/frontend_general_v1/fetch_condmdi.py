"""Fetch one official inference archive; verify before exposing it as complete."""
import asyncio
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "text_motion"
URL = ("https://drive.usercontent.google.com/download?"
       "id=1aP-z1JxSCTcUHhMqqdL2wbwQJUZWHT2j&export=download&confirm=t")
EXPECTED_BYTES = 1789794198


def save(data):
    tmp = OUT / "download.writing"
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(OUT / "download.json")


async def main():
    OUT.mkdir(exist_ok=True)
    archive = OUT / "condmdi_random_joints.zip"
    partial = archive.with_suffix(".zip.part")
    assert not archive.exists(), "Existing completed archive must not be overwritten"
    assert not partial.exists(), "Inspect existing partial download before restarting"
    assert shutil.disk_usage(OUT).free > EXPECTED_BYTES + 5 * 1024**3
    state = dict(state="downloading", url=URL, expected_bytes=EXPECTED_BYTES,
                 started=time.time(), archive=str(archive))
    with (OUT / "download.log").open("x") as log:
        proc = await asyncio.create_subprocess_exec(
            "curl", "--proxy", "http://127.0.0.1:17904", "-fL", "--retry", "2",
            "--connect-timeout", "15", "--max-time", "7200",
            "--speed-limit", "1024", "--speed-time", "180",
            "-o", str(partial), URL, stdout=log, stderr=log)
        state["pid"] = proc.pid
        save(state)
        state["exit_code"] = await proc.wait()
    if state["exit_code"]:
        state.update(state="failed", finished=time.time())
        save(state)
        raise RuntimeError("Official archive download failed; inspect log")
    assert partial.stat().st_size == EXPECTED_BYTES
    with zipfile.ZipFile(partial) as z:
        assert z.testzip() is None
        members = z.infolist()
        assert all(not Path(m.filename).is_absolute()
                   and ".." not in Path(m.filename).parts for m in members)
        state["members"] = [dict(name=m.filename, bytes=m.file_size) for m in members]
    with partial.open("rb") as stream:
        state["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    partial.rename(archive)
    state.update(state="downloaded_crc_verified_not_loaded", finished=time.time(),
                 integrity_scope="Official HTTPS source, local SHA256 and ZIP CRC; no independently published checksum.")
    save(state)


if __name__ == "__main__":
    asyncio.run(main())
