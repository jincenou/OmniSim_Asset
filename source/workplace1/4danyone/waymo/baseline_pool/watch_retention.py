"""Recheck completed empty-mask retention failures while preparation continues."""
import argparse
import asyncio
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-pid", type=int, nargs="+", required=True)
    args = parser.parse_args()
    while True:
        child = await asyncio.create_subprocess_exec(
            "/mnt/ojc/miniconda3/envs/4danyone/bin/python", "-u", str(ROOT / "repair_empty_retention.py"),
            env=dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2"))
        code = await child.wait()
        if code:
            raise RuntimeError(f"Retention audit exited {code}; inspect before retry")
        live = False
        for pid in args.parent_pid:
            try:
                os.kill(pid, 0)
                command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                if any(Path(os.fsdecode(arg)).name in
                       {"run_native_pool.py", "run_predicted_pool.py"} for arg in command if arg):
                    live = True
            except (ProcessLookupError, FileNotFoundError):
                pass
        if not live:
            break
        await asyncio.sleep(45)


if __name__ == "__main__":
    asyncio.run(main())
