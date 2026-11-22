#!/usr/bin/env bash
set -uo pipefail
source /mnt/ojc/workplace1/4danyone/scripts/env.sh
unset PYTORCH_CUDA_ALLOC_CONF
export CUDA_DEVICE_ORDER=PCI_BUS_ID
/mnt/ojc/miniconda3/envs/4danyone/bin/python /mnt/ojc/workplace1/4danyone/waymo/round2/scripts/after_batch.py >> /mnt/ojc/workplace1/4danyone/waymo/round2/logs/after_batch.log 2>&1
result=$?
printf '%s\n' "$result" > /mnt/ojc/workplace1/4danyone/waymo/round2/logs/after_batch.exit
exit "$result"
