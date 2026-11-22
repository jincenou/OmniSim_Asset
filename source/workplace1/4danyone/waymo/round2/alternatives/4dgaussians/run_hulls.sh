#!/usr/bin/env bash
set -uo pipefail
project=/mnt/ojc/workplace1/4danyone/waymo/round2/alternatives/4dgaussians
/mnt/ojc/miniconda3/envs/4dgaussians/bin/python "$project/hull_jobs.py" > "$project/logs/hull-controller.log" 2>&1
hull_exit=$?
printf '%s\n' "$hull_exit" > "$project/logs/hull-controller.exit"
exit "$hull_exit"
