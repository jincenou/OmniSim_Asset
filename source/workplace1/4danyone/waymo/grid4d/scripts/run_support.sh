#!/usr/bin/env bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/waymo/grid4d/scripts/env.sh
exec python -u /mnt/ojc/workplace1/4danyone/waymo/grid4d/scripts/support_queue.py > /mnt/ojc/workplace1/4danyone/waymo/grid4d/logs/support_queue.log 2>&1
