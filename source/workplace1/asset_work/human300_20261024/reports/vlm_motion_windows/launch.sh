#!/bin/bash
set -euo pipefail
export HUMAN_RUN_ROOT=/mnt/ojc/workplace1/asset_work/human300_20261024
export HUMAN_ASSET_ROOT=/mnt/ojc/workplace1/asset_library/human
cd /mnt/ojc/workplace1/human
exec /mnt/ojc/miniconda3/envs/vllm_vlm/bin/python -u motion_review.py H0107 H0001 H0144 --window-probe --gpu 2 > /mnt/ojc/workplace1/asset_work/human300_20261024/reports/vlm_motion_windows/run.log 2>&1
