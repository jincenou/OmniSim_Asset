#!/usr/bin/env bash
set -uo pipefail
project=/mnt/ojc/workplace1/4danyone/waymo/round2/alternatives/4dgaussians
/mnt/ojc/miniconda3/envs/4dgaussians/bin/python "$project/native_validation_jobs.py" > "$project/logs/native-validation-controller.log" 2>&1
validation_exit=$?
printf '%s\n' "$validation_exit" > "$project/logs/native-validation-controller.exit"
exit "$validation_exit"
