#!/usr/bin/env bash
set -uo pipefail
source /mnt/ojc/workplace1/4danyone/scripts/env.sh
export CUDA_VISIBLE_DEVICES=''
/mnt/ojc/miniconda3/envs/4danyone/bin/python /mnt/ojc/workplace1/4danyone/waymo/round2/scripts/native_master.py >> /mnt/ojc/workplace1/4danyone/waymo/round2/logs/native_master.log 2>&1
result=$?
printf '%s\n' "$result" > /mnt/ojc/workplace1/4danyone/waymo/round2/logs/native_master.exit
exit "$result"
