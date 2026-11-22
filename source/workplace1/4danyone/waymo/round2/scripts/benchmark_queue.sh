#!/usr/bin/env bash
set -uo pipefail
source /mnt/ojc/workplace1/4danyone/scripts/env.sh
/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python /mnt/ojc/workplace1/4danyone/waymo/round2/scripts/benchmark_queue.py > /mnt/ojc/workplace1/4danyone/waymo/round2/benchmark/logs/controller.log 2>&1
result=$?
printf '%s\n' "$result" > /mnt/ojc/workplace1/4danyone/waymo/round2/benchmark/logs/controller.exit
exit "$result"
