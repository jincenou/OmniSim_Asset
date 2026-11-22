#!/bin/bash
set -euo pipefail
export HUMAN_RUN_ROOT=/mnt/ojc/workplace1/asset_work/human300_20261024
export HUMAN_ASSET_ROOT=/mnt/ojc/workplace1/asset_library/human
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd /mnt/ojc/workplace1/human
exec /mnt/ojc/miniconda3/envs/4danyone/bin/python -u playback.py --all-complete > /mnt/ojc/workplace1/asset_work/human300_20261024/reports/practical_playback/backfill.log 2>&1
