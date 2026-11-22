#!/bin/bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/eff4d/scheduler/env.sh
trap 'code=$?; echo "$code" > "$EFF4D_ROOT/scheduler/smoke.exit"' EXIT
export CUDA_VISIBLE_DEVICES=4
python "$EFF4D_ROOT/adapters/train.py" --object R07 --protocol S0 --variant E0 --run-id S0_smoke150_v1 --steps 150
