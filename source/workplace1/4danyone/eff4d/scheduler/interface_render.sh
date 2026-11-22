#!/bin/bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/eff4d/scheduler/env.sh
trap 'code=$?; echo "$code" > "$EFF4D_ROOT/scheduler/interface_render.exit"' EXIT
export CUDA_VISIBLE_DEVICES=4
python "$EFF4D_ROOT/adapters/render_evaluate.py" --checkpoint "$EFF4D_ROOT/research/interface_test/short_clip/05_reconstruction/interface20_v1/checkpoint.pt" --demo-only
