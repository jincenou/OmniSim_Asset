#!/usr/bin/env bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/waymo/grid4d/scripts/env.sh
export CUDA_VISIBLE_DEVICES=''
exec python -u /mnt/ojc/workplace1/4danyone/waymo/grid4d/scripts/historical_exposure_queue.py > /mnt/ojc/workplace1/4danyone/waymo/grid4d/logs/historical_exposure_queue.log 2>&1
