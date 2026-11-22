#!/usr/bin/env bash
set -uo pipefail
cd /mnt/ojc/workplace1/4danyone/waymo/round2/alternatives/4dgaussians
/mnt/ojc/miniconda3/envs/4dgaussians/bin/python -u finish_short_demos.py > logs/short_demo_controller.log 2>&1
code=$?
printf '%s\n' "$code" > logs/short_demo_controller.exit
exit "$code"
