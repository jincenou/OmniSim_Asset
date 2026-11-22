#!/bin/bash
set -euo pipefail
export HUMAN_RUN_ROOT=/mnt/ojc/workplace1/asset_work/human300_20261024/checks/masks
exec /mnt/ojc/miniconda3/envs/4danyone/bin/python -u /mnt/ojc/workplace1/human/run.py batch masks_check N04 --gpus 4 5 6 7
