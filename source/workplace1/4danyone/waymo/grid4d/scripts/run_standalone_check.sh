#!/usr/bin/env bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/waymo/grid4d/scripts/env.sh
export CUDA_VISIBLE_DEVICES=GPU-cd4b2342-fba4-2b42-beb9-60b698debc2d
exec python -u /mnt/ojc/workplace1/4danyone/waymo/grid4d/scripts/replay_asset.py --run /mnt/ojc/workplace1/4danyone/waymo/grid4d/runs/N04_33k_v1 --iteration 33000 --view 1 --output /mnt/ojc/workplace1/4danyone/waymo/grid4d/research/standalone/N04_view01.mp4 > /mnt/ojc/workplace1/4danyone/waymo/grid4d/logs/standalone_check.log 2>&1
