#!/bin/bash
set -euo pipefail
export HUMAN_RUN_ROOT=/mnt/ojc/workplace1/asset_work/human_quality_repair_20261026
export HUMAN_ASSET_ROOT="$HUMAN_RUN_ROOT/candidates"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/mnt/ojc/miniconda3/envs/4danyone/bin/python -u /mnt/ojc/workplace1/human/frontend.py subset H0384 --source-root /mnt/ojc/workplace1/asset_work/human300_20261024 --source-identity H0384 --end-frame 56
/mnt/ojc/miniconda3/envs/4danyone/bin/python -u /mnt/ojc/workplace1/human/frontend.py subset H0773 --source-root /mnt/ojc/workplace1/asset_work/human300_20261024 --source-identity H0773 --end-frame 56
/mnt/ojc/miniconda3/envs/4danyone/bin/python -u /mnt/ojc/workplace1/human/frontend.py subset H0787 --source-root /mnt/ojc/workplace1/asset_work/human300_20261024 --source-identity H0787 --end-frame 53
exec /mnt/ojc/miniconda3/envs/4danyone/bin/python -u /mnt/ojc/workplace1/human/run.py batch repair --cases H0384 H0773 H0787 H0038 --gpus 0 1 2 3 4 5 6 7 --min-free-mib 26000
