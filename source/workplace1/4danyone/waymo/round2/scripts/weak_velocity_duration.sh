#!/usr/bin/env bash
set -uo pipefail
source /mnt/ojc/workplace1/4danyone/scripts/env.sh
unset PYTORCH_CUDA_ALLOC_CONF
/mnt/ojc/miniconda3/envs/4danyone/bin/python /mnt/ojc/workplace1/4danyone/waymo/round2/scripts/weak_velocity_queue.py --sigma8 >> /mnt/ojc/workplace1/4danyone/waymo/round2/logs/weak_velocity_duration_queue.log 2>&1
result=$?
printf '%s\n' "$result" > /mnt/ojc/workplace1/4danyone/waymo/round2/logs/weak_velocity_duration_queue.exit
exit "$result"
