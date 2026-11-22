#!/usr/bin/env bash
set -euo pipefail
R=/mnt/ojc/workplace1/4danyone/waymo/round2
trap 'printf "%s\n" "$?" > "$R/logs/r07_audit.exit"' EXIT
exec > "$R/logs/r07_audit.log" 2>&1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python "$R/scripts/raw_camera_audit.py" --review-id R07
/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python "$R/scripts/r07_prepare.py"
