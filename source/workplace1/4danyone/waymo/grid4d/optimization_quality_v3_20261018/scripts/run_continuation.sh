#!/usr/bin/env bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/waymo/grid4d/optimization_quality_v3_20261018/scripts/env.sh
exec "$GRID4D_ENV/bin/python" -u "$GRID4D_QUALITY/scripts/continuation_queue.py" > "$GRID4D_QUALITY/logs/continuation_queue.log" 2>&1
