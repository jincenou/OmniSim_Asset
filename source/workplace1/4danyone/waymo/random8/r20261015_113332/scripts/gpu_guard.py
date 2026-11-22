"""Check the CUDA runtime UUID on this older torch build before GPU work."""
import ctypes
import os
import subprocess
import uuid
import torch
import asyncio
import fcntl
import json
import sys
from pathlib import Path

_ORIGINAL_LANE = None


async def _priority_lane():
    """Three priority lanes plus at most one original optimization process.

    Already-running jobs retain their imported code. This admission check only
    applies to future original-queue children, before CUDA initialization.
    """
    root = Path(__file__).resolve().parents[1]
    # Budget rendering spawns a same-device child while its parent waits.
    # Device leases already protect rendering; this extra lane only bounds
    # independent optimization jobs, so evaluators must not acquire it.
    if Path(sys.argv[0]).stem.startswith("evaluate_branch"):
        return None
    priority = root / "FIRST4_STATUS.json"
    if not priority.exists() or json.loads(priority.read_text()).get("state") != "first_four_priority_active":
        return None
    handle = (root / "first4_original_gpu_lane.lock").open("a")
    record = root / "research" / f"original_gpu_lane_{os.getpid()}.json"
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            record.write_text(json.dumps({"state": "waiting_original_GPU_lane", "pid": os.getpid(), "physical_gpu": int(os.environ["CUDA_VISIBLE_DEVICES"])}))
            await asyncio.sleep(2)
        else:
            record.write_text(json.dumps({"state": "original_GPU_lane_acquired", "pid": os.getpid(), "physical_gpu": int(os.environ["CUDA_VISIBLE_DEVICES"])}))
            return handle


def validate_gpu():
    global _ORIGINAL_LANE
    _ORIGINAL_LANE = asyncio.run(_priority_lane())
    physical = int(os.environ["CUDA_VISIBLE_DEVICES"])
    assert physical in (4, 5, 6, 7)
    expected = subprocess.check_output(["nvidia-smi", f"--id={physical}", "--query-gpu=uuid",
        "--format=csv,noheader"], text=True).strip().removeprefix("GPU-")
    torch.cuda.init()
    assert torch.cuda.device_count() == 1
    # Read the device in the active torch context; driver ordinals alone would
    # not establish CUDA_VISIBLE_DEVICES mapping.
    torch.empty(1, device="cuda")
    driver = ctypes.CDLL("libcuda.so.1")
    context_device = ctypes.c_int()
    driver.cuCtxGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuCtxGetDevice.restype = ctypes.c_int
    assert driver.cuCtxGetDevice(ctypes.byref(context_device)) == 0
    value = (ctypes.c_ubyte * 16)()
    driver.cuDeviceGetUuid.argtypes = [ctypes.c_void_p, ctypes.c_int]
    driver.cuDeviceGetUuid.restype = ctypes.c_int
    assert driver.cuDeviceGetUuid(ctypes.byref(value), context_device.value) == 0
    actual = str(uuid.UUID(bytes=bytes(value)))
    assert actual == expected, (physical, actual, expected)
    return {"physical_gpu": physical, "cuda_uuid": actual, "pid": os.getpid(), "torch": torch.__version__}
