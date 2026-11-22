#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
cd /mnt/ojc/workplace1/asset_work/human_motion_20261025/code
exec /mnt/ojc/miniconda3/envs/4danyone/bin/python -u motion_experiment.py run >> /mnt/ojc/workplace1/asset_work/human_motion_20261025/logs/controller.log 2>&1
