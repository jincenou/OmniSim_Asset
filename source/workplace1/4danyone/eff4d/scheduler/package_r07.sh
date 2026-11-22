#!/bin/bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/eff4d/scheduler/env.sh
trap 'code=$?; echo "$code" > "$EFF4D_ROOT/scheduler/package_r07.exit"' EXIT
export CUDA_VISIBLE_DEVICES=''
python "$EFF4D_ROOT/adapters/package.py" --object R07
