#!/usr/bin/env bash
set -uo pipefail
project=/mnt/ojc/workplace1/4danyone/waymo/round2/alternatives/4dgaussians
/mnt/ojc/miniconda3/envs/4dgaussians/bin/python "$project/interpolation_jobs.py" > "$project/logs/interpolation-controller.log" 2>&1
interpolation_exit=$?
printf '%s\n' "$interpolation_exit" > "$project/logs/interpolation-controller.exit"
exit "$interpolation_exit"
