#!/usr/bin/env bash
set -euo pipefail
R=/mnt/ojc/workplace1/4danyone/waymo/round2
trap 'printf "%s\n" "$?" > "$R/logs/moving_candidates_v2.exit"' EXIT
exec > "$R/logs/moving_candidates_v2.log" 2>&1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python "$R/scripts/moving_candidates_v2.py"
