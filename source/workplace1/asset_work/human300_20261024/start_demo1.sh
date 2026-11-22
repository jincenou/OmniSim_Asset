#!/bin/bash
set -euo pipefail
export HUMAN_RUN_ROOT=/mnt/ojc/workplace1/asset_work/human300_20261024
exec /mnt/ojc/miniconda3/envs/4danyone/bin/python -u /mnt/ojc/workplace1/human/run.py frontend --cases H0001 --gpus 0
