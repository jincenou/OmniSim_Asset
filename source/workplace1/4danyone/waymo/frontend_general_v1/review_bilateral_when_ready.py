import asyncio
import json
from pathlib import Path

G = Path(__file__).resolve().parent
E = G / "text_motion/bilateral_context/repair_v1"


async def main():
    while True:
        state = json.loads((E / "status.json").read_text())["state"]
        if state == "completed_needs_motion_review":
            break
        if state in ("failed", "preparation_failed"):
            raise RuntimeError("Incomplete paired experiment; do not publish comparisons")
        await asyncio.sleep(5)
    with (E / "logs/review.log").open("x") as log:
        p = await asyncio.create_subprocess_exec(
            "/mnt/ojc/miniconda3/envs/condmdi/bin/python", str(G / "review_bilateral_motion.py"),
            stdout=log, stderr=log)
        (E / "review_status.json").write_text(json.dumps(dict(state="running", pid=p.pid)))
        rc = await p.wait()
    (E / "review_status.json").write_text(json.dumps(
        dict(state="review_ready" if rc == 0 else "failed", exit_code=rc, quality_approved=False), indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
