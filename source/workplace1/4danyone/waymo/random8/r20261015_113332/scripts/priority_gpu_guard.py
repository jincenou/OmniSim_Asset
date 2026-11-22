"""Check the CUDA runtime UUID on this older torch build before GPU work."""
import ctypes
import os
import subprocess
import uuid
import torch


def validate_gpu():
    physical = int(os.environ["CUDA_VISIBLE_DEVICES"])
    assert physical in (1, 6, 7)
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
