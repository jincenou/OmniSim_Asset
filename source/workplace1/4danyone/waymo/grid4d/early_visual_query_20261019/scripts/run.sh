#!/usr/bin/env bash
set -euo pipefail
cd /mnt/ojc/workplace1/4danyone/waymo/grid4d/early_visual_query_20261019
source scripts/env.sh
script="$1"
shift
exec /mnt/ojc/miniconda3/envs/grid4d/bin/python "scripts/$script.py" "$@"
