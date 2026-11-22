#!/bin/bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/eff4d/scheduler/env.sh
trap 'code=$?; echo "$code" > "$EFF4D_ROOT/scheduler/confidence_r07.exit"' EXIT
export CUDA_VISIBLE_DEVICES=5
python "$EFF4D_ROOT/adapters/confidence.py" --object R07 --protocol S1
