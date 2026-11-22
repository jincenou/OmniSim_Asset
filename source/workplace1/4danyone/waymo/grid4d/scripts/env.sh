#!/usr/bin/env bash
export GRID4D_ROOT=/mnt/ojc/workplace1/4danyone/waymo/grid4d
export GRID4D_ENV=/mnt/ojc/miniconda3/envs/grid4d
export CUDA_HOME=/mnt/ojc/miniconda3/envs/4dgaussians
export PATH="$GRID4D_ENV/bin:$CUDA_HOME/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CC=/usr/bin/gcc-9 CXX=/usr/bin/g++-9
export PIP_CACHE_DIR="$GRID4D_ROOT/cache/pip" TMPDIR="$GRID4D_ROOT/tmp"
export TORCH_EXTENSIONS_DIR="$GRID4D_ROOT/cache/extensions" GRID4D_HASH_BUILD="$GRID4D_ROOT/cache/extensions/hashencoder"
export TORCH_HOME="$GRID4D_ROOT/cache/torch" MPLCONFIGDIR="$GRID4D_ROOT/cache/matplotlib"
export XDG_CACHE_HOME="$GRID4D_ROOT/cache/xdg" CUDA_CACHE_PATH="$GRID4D_ROOT/cache/cuda"
export NUMBA_CACHE_DIR="$GRID4D_ROOT/cache/numba" CUDA_DEVICE_ORDER=PCI_BUS_ID
