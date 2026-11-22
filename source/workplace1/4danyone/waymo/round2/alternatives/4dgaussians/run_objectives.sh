#!/usr/bin/env bash
set -uo pipefail
cd /mnt/ojc/workplace1/4danyone/waymo/round2/alternatives/4dgaussians
/mnt/ojc/miniconda3/envs/4dgaussians/bin/python -u objective_jobs.py > logs/objective_controller.log 2>&1
code=$?
printf '%s\n' "$code" > logs/objective_controller.exit
exit "$code"
