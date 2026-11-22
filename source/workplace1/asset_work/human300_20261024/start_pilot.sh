#!/bin/bash
set -euo pipefail
export HUMAN_RUN_ROOT=/mnt/ojc/workplace1/asset_work/human300_20261024
export HUMAN_ASSET_ROOT=/mnt/ojc/workplace1/asset_library/human
exec /mnt/ojc/miniconda3/envs/4danyone/bin/python -u /mnt/ojc/workplace1/human/run.py batch pilot --cases H0001 H0002 --gpus 0 2
