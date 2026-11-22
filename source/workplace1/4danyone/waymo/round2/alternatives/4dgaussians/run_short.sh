#!/usr/bin/env bash
set -uo pipefail
cd /mnt/ojc/workplace1/4danyone/waymo/round2/alternatives/4dgaussians
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
/mnt/ojc/miniconda3/envs/4dgaussians/bin/python -u short_jobs.py > logs/short_controller.log 2>&1
code=$?
printf '%s\n' "$code" > logs/short_controller.exit
exit "$code"
