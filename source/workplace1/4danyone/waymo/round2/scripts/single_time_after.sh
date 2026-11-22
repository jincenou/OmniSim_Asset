#!/usr/bin/env bash
set -euo pipefail
R=/mnt/ojc/workplace1/4danyone/waymo/round2
trap 'printf "%s\n" "$?" > "$R/logs/single_time_after.exit"' EXIT
exec > "$R/logs/single_time_after.log" 2>&1
/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python "$R/scripts/single_time_after.py"
