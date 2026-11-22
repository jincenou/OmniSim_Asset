#!/usr/bin/env bash
set -uo pipefail
project=/mnt/ojc/workplace1/4danyone/waymo/round2/alternatives/4dgaussians
/mnt/ojc/miniconda3/envs/4dgaussians/bin/python "$project/profile_jobs.py" > "$project/logs/profile-controller.log" 2>&1
profile_exit=$?
printf '%s\n' "$profile_exit" > "$project/logs/profile-controller.exit"
exit "$profile_exit"
