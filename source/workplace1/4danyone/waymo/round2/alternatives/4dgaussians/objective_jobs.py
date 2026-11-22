"""Wait on live official/alpha controllers, then create independent paired comparisons."""
import asyncio
from argparse import ArgumentParser
from datetime import datetime
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

A = Path(__file__).resolve().parent
PATH = A / "objective_queue.json"
STATE = {"pid": os.getpid(), "state": "waiting_for_live_training", "jobs": {}}


def write():
    STATE["updated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    temporary = PATH.with_suffix(".writing")
    temporary.write_text(json.dumps(STATE, indent=2) + "\n")
    temporary.replace(PATH)


async def one(prefix):
    sources = [(prefix + "_recipe30k_v1", "short_queue.json", b"short_jobs.py"),
        (prefix + "_alpha_recipe30k_v1", "alpha_queue.json", b"alpha_jobs.py")]
    STATE["jobs"][prefix] = {"state": "waiting_for_existing_results"}
    write()
    while True:
        ready = True
        for run, queue_name, identity in sources:
            source = A / "runs" / run / "evaluation/results.json"
            if source.exists():
                assert json.loads(source.read_text())["state"] == "new_process_replayed_evaluated_decoded_visual_review_pending"
                continue
            ready = False
            queue = json.loads((A / queue_name).read_text())
            command_path = Path(f"/proc/{queue['pid']}/cmdline")
            assert command_path.exists() and identity in command_path.read_bytes(), "Source controller no longer live"
            assert queue["state"] != "failed", "Source controller failed"
        if ready:
            break
        await asyncio.sleep(30)
    cmd = ["/mnt/ojc/miniconda3/envs/4dgaussians/bin/python", str(A / "compare_objectives.py"), "--prefix", prefix]
    suffix = "" if STATE["attempt"] == 1 else f"_r{STATE['attempt']}"
    logpath = A / "logs" / (prefix + "_objective_comparison" + suffix + ".log")
    with logpath.open("xb") as log:
        process = await asyncio.create_subprocess_exec(*cmd, cwd=A, env=dict(os.environ,
            OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4"), stdout=log, stderr=asyncio.subprocess.STDOUT)
        STATE["jobs"][prefix].update(state="running", pid=process.pid, command=cmd, log=str(logpath))
        write()
        code = await process.wait()
    (A / "logs" / (prefix + "_objective_comparison" + suffix + ".exit")).write_text(str(code) + "\n")
    STATE["jobs"][prefix].update(state="compared_decoded_review_pending" if code == 0 else "failed", exit_code=code)
    write()
    if code:
        raise RuntimeError(f"Comparison{prefix}failed:exit{code}")


async def main():
    global PATH
    parser = ArgumentParser()
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    assert args.attempt > 0
    STATE["attempt"] = args.attempt
    if args.attempt > 1:
        PATH = A / f"objective_queue_r{args.attempt}.json"
    assert not PATH.exists()
    write()
    values = await asyncio.gather(one("m02_61"), one("r07_61"), return_exceptions=True)
    errors = [str(value) for value in values if isinstance(value, Exception)]
    STATE.update(state="failed" if errors else "paired_alpha_results_compared_decoded_visual_review_pending", failures=errors)
    write()
    if errors:
        raise RuntimeError("; ".join(errors))


if __name__ == "__main__":
    asyncio.run(main())
