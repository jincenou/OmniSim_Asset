#!/usr/bin/env bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/waymo/grid4d/optimization_quality_v3_20261018/scripts/env.sh
export CUDA_VISIBLE_DEVICES=GPU-f9cfc0e8-a5ac-7168-2700-4b52fee1e21d
export PYTHONPATH="$GRID4D_ROOT/optimization_boundary_20261018/scripts:$GRID4D_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
"$GRID4D_ENV/bin/python" -u "$GRID4D_QUALITY/source/train_bmatch.py" --label N04 --arm B > "$GRID4D_QUALITY/logs/N04_Bmatch.log" 2>&1
"$GRID4D_ENV/bin/python" -c 'from pathlib import Path; Path("/mnt/ojc/workplace1/4danyone/waymo/grid4d/optimization_quality_v3_20261018/logs/Bmatch.complete").touch()'
