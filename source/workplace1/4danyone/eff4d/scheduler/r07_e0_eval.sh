#!/bin/bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/eff4d/scheduler/env.sh
trap 'code=$?; echo "$code" > "$EFF4D_ROOT/scheduler/r07_e0_eval.exit"' EXIT
export CUDA_VISIBLE_DEVICES=4
python "$EFF4D_ROOT/scheduler/admission.py" --gpu 4 --min-free-mib 4000
python "$EFF4D_ROOT/adapters/render_evaluate.py" --checkpoint "$EFF4D_ROOT/objects/OBJ_0018/clips/C001/05_reconstruction/E0_S1_12k_v1/checkpoint.pt"
