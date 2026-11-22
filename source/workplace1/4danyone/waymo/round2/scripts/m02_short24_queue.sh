#!/usr/bin/env bash
set -uo pipefail
source /mnt/ojc/workplace1/4danyone/scripts/env.sh
unset PYTORCH_CUDA_ALLOC_CONF
/mnt/ojc/miniconda3/envs/4danyone/bin/python /mnt/ojc/workplace1/4danyone/waymo/round2/scripts/short24_queue.py --plan /mnt/ojc/workplace1/4danyone/waymo/round2/m02_short24_plan.json --state /mnt/ojc/workplace1/4danyone/waymo/round2/m02_short24_queue.json >> /mnt/ojc/workplace1/4danyone/waymo/round2/logs/m02_short24_queue.log 2>&1
result=$?
printf '%s\n' "$result" > /mnt/ojc/workplace1/4danyone/waymo/round2/logs/m02_short24_queue.exit
exit "$result"
