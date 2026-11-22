#!/bin/bash
set -euo pipefail
export HUMAN_RUN_ROOT=/mnt/ojc/workplace1/asset_work/human300_20261024
export HUMAN_ASSET_ROOT=/mnt/ojc/workplace1/asset_library/human
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
/mnt/ojc/miniconda3/envs/4danyone/bin/python -u /mnt/ojc/workplace1/human/placement.py --all-complete --concurrency 2
/mnt/ojc/miniconda3/envs/4danyone/bin/python -u /mnt/ojc/workplace1/human/placement.py --all-complete --verify-gpus 1 2
