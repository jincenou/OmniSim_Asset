#!/usr/bin/env bash
source /mnt/ojc/workplace1/4danyone/waymo/grid4d/scripts/env.sh
export GRID4D_BOUNDARY=/mnt/ojc/workplace1/4danyone/waymo/grid4d/optimization_boundary_20261018
export TMPDIR="$GRID4D_BOUNDARY/tmp"
export MPLCONFIGDIR="$GRID4D_BOUNDARY/cache/matplotlib"
export CUDA_CACHE_PATH="$GRID4D_BOUNDARY/cache/cuda"
export NUMBA_CACHE_DIR="$GRID4D_BOUNDARY/cache/numba"
