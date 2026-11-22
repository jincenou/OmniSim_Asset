"""Explicit repair of only N04 grid61's interrupted 15k-budget evaluation."""
import asyncio
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import controller_v3 as queue

R = Path(__file__).resolve().parents[1]
RECEIPT = R / "research/N04_4dgs_grid61_evaluation_repair.json"
OUTPUT = R / "samples/N04/evaluations/4dgs_grid61_snapshot12000/results.json"
MAIN = R / "samples/N04/evaluations/4dgs_grid61/results.json"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


async def main():
    assert not RECEIPT.exists(), "Explicit single repair; do not silently retry"
    assert not OUTPUT.parent.exists(), "Preserve any existing attempt before a new repair"
    main_hash = hashlib.sha256(MAIN.read_bytes()).hexdigest()
    record = {"state": "waiting_GPU5_lease_for_budget_evaluation_only", "controller_pid": __import__("os").getpid(),
              "scope": "Only total15k=fine12000 snapshot evaluation. Final model/main metrics preserved; no retraining.",
              "original_stage_exit_code": 1, "started_at": now(), "main_metrics_sha256": main_hash}
    queue.write(RECEIPT, record)
    async with queue.devices("N04/grid61_budget_repair", [5], 10000) as handles:
        command = [queue.PY["hust"], "-u", str(R / "scripts/evaluate_branch.py"), "--label", "N04",
                   "--branch", "4dgs_grid61", "--snapshot", "12000"]
        log = R / "logs/N04_grid61_budget_repair_attempt2.log"
        with log.open("xb") as stream:
            process = await asyncio.create_subprocess_exec(*command, cwd=R, env=queue.env_for("N04", [5]),
                stdout=stream, stderr=asyncio.subprocess.STDOUT, start_new_session=True, pass_fds=handles)
            record.update(state="running", pid=process.pid, process_group=process.pid, physical_gpus=[5],
                          command=command, log=str(log))
            queue.write(RECEIPT, record)
            code = await process.wait()
        record.update(state="complete" if code == 0 else "failed", exit_code=code, ended_at=now())
        if code == 0:
            result = json.loads(OUTPUT.read_text())
            assert result["state"] == "new_process_replayed_evaluated_decoded_visual_review_pending"
            assert hashlib.sha256(MAIN.read_bytes()).hexdigest() == main_hash
            record.update(proof=str(OUTPUT), repaired_budget_mean=result["mean"],
                          main_metrics_unchanged=True, device=result["device"])
        queue.write(RECEIPT, record)
        if code:
            raise RuntimeError(f"Explicit budget repair failed with {code}; no automatic retry")


if __name__ == "__main__":
    asyncio.run(main())
