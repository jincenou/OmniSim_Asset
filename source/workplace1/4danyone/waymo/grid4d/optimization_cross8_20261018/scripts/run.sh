#!/usr/bin/env bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/waymo/grid4d/optimization_cross8_20261018/scripts/env.sh
exec "$GRID4D_ENV/bin/python" -u "$GRID4D_QUALITY/scripts/run_cross8.py" >> "$GRID4D_QUALITY/logs/controller.log" 2>&1
