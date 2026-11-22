#!/bin/bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/eff4d/scheduler/env.sh
trap 'code=$?; echo "$code" > "$EFF4D_ROOT/scheduler/m02_e1.exit"' EXIT
export CUDA_VISIBLE_DEVICES=7
python "$EFF4D_ROOT/scheduler/admission.py" --gpu 7 --min-free-mib 6000
python "$EFF4D_ROOT/adapters/train.py" --object M02 --protocol S1 --variant E1 --run-id E1_S1_12k_v1 --steps 12000
