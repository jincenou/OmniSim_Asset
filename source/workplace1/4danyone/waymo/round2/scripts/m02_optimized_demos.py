"""Produce viewing packages for the two already-trained M02 configurations."""
import asyncio
import json
import os
from pathlib import Path
import shutil
from after_batch import ROUND, W, now, write
from resource_lease import lease, wait_memory

STATE = {"state": "starting", "pid": os.getpid(), "jobs": {}, "started_at": now()}
PATH = ROUND / "m02_optimized_demos.json"


async def one(variant, gpu):
    async with lease([gpu]) as handles:
        await wait_memory([gpu], 6000)
        assert shutil.disk_usage(ROUND).free > 35 * 2**30
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
            CUDA_HOME="/mnt/ojc/miniconda3/envs/asset_harvester_2026",
            TORCH_EXTENSIONS_DIR=str(W / "reconstruction/deps/torch_extensions"),
            TORCH_CUDA_ARCH_LIST="8.9", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
        env["PATH"] = env["CUDA_HOME"] + "/bin:" + env["PATH"]
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        cmd = ["/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python", str(ROUND / "scripts/render_demo61.py"),
            "--job", str(ROUND / "demo_jobs" / f"m02_short61_{variant}.json")]
        with (ROUND / "logs" / f"m02_optimized_{variant}.log").open("xb") as log:
            child = await asyncio.create_subprocess_exec(*cmd, cwd=ROUND / "backend", env=env,
                stdout=log, stderr=asyncio.subprocess.STDOUT, pass_fds=handles)
            STATE["jobs"][variant] = {"state": "running", "gpu": gpu, "pid": child.pid, "command": cmd}
            write(PATH, STATE)
            code = await child.wait()
        (ROUND / "logs" / f"m02_optimized_{variant}.exit").write_text(str(code) + "\n")
        STATE["jobs"][variant].update(state="rendered_decoded_review_pending" if not code else "failed", exit_code=code)
        write(PATH, STATE)
        if code:
            raise RuntimeError(f"M02{variant} rendering failed: exit{code}")


async def main():
    assert not PATH.exists()
    write(PATH, STATE)
    results = await asyncio.gather(one("sigma15", 4), one("sigma8", 5), return_exceptions=True)
    errors = [str(result) for result in results if isinstance(result, Exception)]
    STATE.update(state="failed" if errors else "rendered_decoded_review_pending", failures=errors, ended_at=now())
    write(PATH, STATE)
    if errors:
        raise RuntimeError("; ".join(errors))


if __name__ == "__main__":
    asyncio.run(main())
