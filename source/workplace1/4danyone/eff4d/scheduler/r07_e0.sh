#!/bin/bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/eff4d/scheduler/env.sh
trap 'code=$?; echo "$code" > "$EFF4D_ROOT/scheduler/r07_e0.exit"' EXIT
export CUDA_VISIBLE_DEVICES=4
python "$EFF4D_ROOT/scheduler/admission.py" --gpu 4 --min-free-mib 6000
python "$EFF4D_ROOT/adapters/train.py" --object R07 --protocol S1 --variant E0 --run-id E0_S1_12k_v1 --steps 12000
