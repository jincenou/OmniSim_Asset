"""Verify logical device0 against its explicitly authorized physical UUID."""
import ctypes
import os
import uuid
import torch


def verify():
    expected = os.environ['CUDA_VISIBLE_DEVICES']
    assert expected.startswith('GPU-') and ',' not in expected
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.empty(1, device='cuda')
    driver = ctypes.CDLL('libcuda.so.1')
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    assert driver.cuInit(0) == 0
    device = ctypes.c_int()
    driver.cuCtxGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuCtxGetDevice.restype = ctypes.c_int
    assert driver.cuCtxGetDevice(ctypes.byref(device)) == 0
    value = (ctypes.c_ubyte * 16)()
    driver.cuDeviceGetUuid.argtypes = [ctypes.c_void_p, ctypes.c_int]
    driver.cuDeviceGetUuid.restype = ctypes.c_int
    assert driver.cuDeviceGetUuid(ctypes.byref(value), device.value) == 0
    actual = 'GPU-' + str(uuid.UUID(bytes=bytes(value)))
    assert actual == expected, (actual, expected)
    return dict(uuid=actual, name=torch.cuda.get_device_name(0), pid=os.getpid(), torch=torch.__version__)
