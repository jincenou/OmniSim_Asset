#!/bin/bash
set -euo pipefail
export HUMAN_RUN_ROOT=/mnt/ojc/workplace1/asset_work/human300_20261024
export HUMAN_ASSET_ROOT=/mnt/ojc/workplace1/asset_library/human
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec /mnt/ojc/miniconda3/envs/4danyone/bin/python -u /mnt/ojc/workplace1/human/run.py batch run --all --gpus 1 0 2 3 4 5 6 7
