#!/usr/bin/env bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/waymo/grid4d/optimization_boundary_20261018/scripts/env.sh
exec "$GRID4D_ENV/bin/python" "$GRID4D_BOUNDARY/scripts/analysis_queue.py" >> "$GRID4D_BOUNDARY/logs/analysis_queue.log" 2>&1
