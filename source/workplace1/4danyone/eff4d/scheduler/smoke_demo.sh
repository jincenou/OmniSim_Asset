#!/bin/bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/eff4d/scheduler/env.sh
trap 'code=$?; echo "$code" > "$EFF4D_ROOT/scheduler/smoke_demo.exit"' EXIT
export CUDA_VISIBLE_DEVICES=7
python "$EFF4D_ROOT/adapters/render_evaluate.py" --checkpoint "$EFF4D_ROOT/objects/OBJ_0018/clips/C001/05_reconstruction/S0_smoke150_v1/checkpoint.pt" --demo-only
