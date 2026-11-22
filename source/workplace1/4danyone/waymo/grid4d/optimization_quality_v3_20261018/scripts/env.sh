#!/usr/bin/env bash
source /mnt/ojc/workplace1/4danyone/waymo/grid4d/scripts/env.sh
export GRID4D_QUALITY=/mnt/ojc/workplace1/4danyone/waymo/grid4d/optimization_quality_v3_20261018
export TMPDIR="$GRID4D_QUALITY/tmp"
export MPLCONFIGDIR="$GRID4D_QUALITY/cache/matplotlib"
export CUDA_CACHE_PATH="$GRID4D_QUALITY/cache/cuda"
export NUMBA_CACHE_DIR="$GRID4D_QUALITY/cache/numba"
