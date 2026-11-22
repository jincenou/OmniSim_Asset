#!/bin/bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/eff4d/scheduler/env.sh
trap 'code=$?; echo "$code" > "$EFF4D_ROOT/scheduler/copy_runtime.exit"' EXIT
export CUDA_VISIBLE_DEVICES=4
/mnt/ojc/miniconda3/bin/python "$EFF4D_ROOT/scheduler/copy_runtime.py"
