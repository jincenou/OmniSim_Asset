#!/usr/bin/env bash
set -uo pipefail
source /mnt/ojc/workplace1/4danyone/scripts/env.sh
unset PYTORCH_CUDA_ALLOC_CONF
/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python /mnt/ojc/workplace1/4danyone/waymo/round2/scripts/budget_queue.py >> /mnt/ojc/workplace1/4danyone/waymo/round2/logs/budget_queue.log 2>&1
result=$?
printf '%s\n' "$result" > /mnt/ojc/workplace1/4danyone/waymo/round2/logs/budget_queue.exit
exit "$result"
