"""Keep the original CPU scheduler from duplicating priority-owned stages.

Its existing GPU child is a separate process group and continues normally.
The scheduler remains its parent, retaining the actual exit status for wait().
After all twelve supplemental terminal evaluations exist, resume that parent
so it collects its child status and adopts the already completed branches.
"""
import asyncio
import fcntl
import json
import os
from pathlib import Path
import signal
from datetime import datetime
from zoneinfo import ZoneInfo

R = Path(__file__).resolve().parents[1]
STATE = R / "FIRST4_HANDOFF.json"
BRANCHES = ("ftgs_base5k", "ftgs_opt15k", "4dgs_grid25", "4dgs_grid61")


def read(path):
    return json.loads(path.read_text())


def own(pid, name):
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
        return str(R).encode() in command and name.encode() in command
    except FileNotFoundError:
        return False


def record(value):
    value["updated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    temporary = STATE.with_suffix(".writing")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(STATE)


async def main():
    lock = (R / "first4_handoff.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    parent = read(R / "STATUS.json")["controller_pid"]
    priority = read(R / "FIRST4_STATUS.json")["controller_pid"]
    assert own(parent, "controller_v3.py") and own(priority, "first4_priority.py")
    state = {"state": "pausing_only_original_cpu_scheduler", "original_controller_pid": parent,
             "priority_controller_pid": priority, "guardian_pid": os.getpid(),
             "scope": "SIGSTOP affects only the original CPU scheduler, never its independently grouped GPU child or any foreign process. The original parent retains actual child exit status.",
             "resume_condition": "All 12 N01-N03 terminal evaluations and 3 completed comparison receipts verified"}
    record(state)
    os.kill(parent, signal.SIGSTOP)
    state["state"] = "original_cpu_scheduler_paused_existing_GPU_child_unaffected"
    while True:
        ready = [f"{label}/{branch}" for label in ("N01", "N02", "N03") for branch in BRANCHES
                 if (R / "samples" / label / "evaluations" / branch / "results.json").exists()]
        comparisons = [label for label in ("N01", "N02", "N03")
                       if (R / "logs" / f"{label}_comparison.json").exists()
                       and read(R / "logs" / f"{label}_comparison.json").get("exit_code") == 0]
        state.update(verified_terminal_evaluations=len(ready), verified_comparisons=comparisons,
                     priority_alive=own(priority, "first4_priority.py"))
        if len(ready) == 12 and len(comparisons) == 3:
            assert own(parent, "controller_v3.py"), "Original scheduler disappeared; do not signal another PID"
            os.kill(parent, signal.SIGCONT)
            state["state"] = "original_scheduler_resumed_adopting_completed_priority_proofs"
            record(state)
            return
        priority_state = read(R / "FIRST4_STATUS.json").get("state", "")
        if not state["priority_alive"] or "terminal" in priority_state:
            state["state"] = "original_cpu_scheduler_kept_paused_priority_failure_requires_review"
        record(state)
        await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(main())
