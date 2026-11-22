import asyncio
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "text_motion"
SHA = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
URL = f"https://openaipublic.azureedge.net/clip/models/{SHA}/ViT-B-32.pt"


async def main():
    dest = ROOT / "clip"
    dest.mkdir(exist_ok=True)
    final = dest / "ViT-B-32.pt"
    partial = dest / "ViT-B-32.pt.part"
    assert not final.exists() and not partial.exists()
    with (dest / "download.log").open("x") as log:
        proc = await asyncio.create_subprocess_exec(
            "curl", "--proxy", "http://127.0.0.1:17904", "-fL", "--retry", "2",
            "--connect-timeout", "15", "--max-time", "3600",
            "-o", str(partial), URL, stdout=log, stderr=log)
        (dest / "status.json").write_text(json.dumps(dict(state="downloading", pid=proc.pid, url=URL)))
        rc = await proc.wait()
    if rc:
        (dest / "status.json").write_text(json.dumps(dict(state="failed", exit_code=rc)))
        raise RuntimeError(rc)
    with partial.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    assert actual == SHA
    partial.rename(final)
    (dest / "status.json").write_text(json.dumps(dict(state="downloaded_official_sha_verified", sha256=actual, bytes=final.stat().st_size), indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
