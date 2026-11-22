#!/usr/bin/env bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/waymo/grid4d/scripts/env.sh
exec python -u /mnt/ojc/workplace1/4danyone/waymo/grid4d/scripts/training_parts_queue.py > /mnt/ojc/workplace1/4danyone/waymo/grid4d/logs/training_parts_queue.log 2>&1
