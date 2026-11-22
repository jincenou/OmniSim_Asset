#!/bin/bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/eff4d/scheduler/env.sh
trap 'code=$?; echo "$code" > "$EFF4D_ROOT/scheduler/confidence_m02.exit"' EXIT
export CUDA_VISIBLE_DEVICES=6
python "$EFF4D_ROOT/scheduler/admission.py" --gpu 6 --min-free-mib 3000
python "$EFF4D_ROOT/adapters/confidence.py" --object M02 --protocol S1
