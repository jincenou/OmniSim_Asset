#!/usr/bin/env bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/waymo/grid4d/query0_fused/scripts/env.sh
cd "$GRID4D_QUERY"
exec "$GRID4D_ENV/bin/python" -u "scripts/$1.py" "${@:2}"
