#!/usr/bin/env bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/waymo/grid4d/optimization_quality_v3_20261018/scripts/env.sh
exec python "$GRID4D_QUALITY/scripts/prefix_queue.py" > "$GRID4D_QUALITY/logs/prefix_queue.log" 2>&1
