#!/usr/bin/env bash
set -uo pipefail
cd /mnt/ojc/workplace1/4danyone/waymo/round2/alternatives/4dgaussians
export PYTHONNOUSERSITE=1
/mnt/ojc/miniconda3/envs/4dgaussians/bin/python -u recover_phone.py > logs/phone_recovery_controller.log 2>&1
code=$?
printf '%s\n' "$code" > logs/phone_recovery_controller.exit
exit "$code"
