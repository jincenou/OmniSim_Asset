#!/usr/bin/env bash
export STUDY_ROOT=/mnt/ojc/workplace1/4danyone/waymo/backend_study
export STUDY_ENV=/mnt/ojc/miniconda3/envs/cem4dgs
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_HOME=/mnt/ojc/miniconda3/envs/4dgaussians
export PATH="$STUDY_ENV/bin:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CC=/usr/bin/gcc-9 CXX=/usr/bin/g++-9
export PIP_CACHE_DIR="$STUDY_ROOT/cache/pip" TMPDIR="$STUDY_ROOT/tmp"
export TORCH_EXTENSIONS_DIR="$STUDY_ROOT/cache/extensions" TORCH_HOME="$STUDY_ROOT/cache/torch"
export MPLCONFIGDIR="$STUDY_ROOT/cache/matplotlib" XDG_CACHE_HOME="$STUDY_ROOT/cache/xdg"
export CUDA_CACHE_PATH="$STUDY_ROOT/cache/cuda" NUMBA_CACHE_DIR="$STUDY_ROOT/cache/numba"
export CUDA_CACHE_MAXSIZE=2147483648
export CUDA_DEVICE_ORDER=PCI_BUS_ID
