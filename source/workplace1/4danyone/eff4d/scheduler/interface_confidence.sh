#!/bin/bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/eff4d/scheduler/env.sh
trap 'code=$?; echo "$code" > "$EFF4D_ROOT/scheduler/interface_confidence.exit"' EXIT
export CUDA_VISIBLE_DEVICES=5
python "$EFF4D_ROOT/adapters/confidence.py" --manifest "$EFF4D_ROOT/research/interface_test/unequal_time_clip/manifest.json" --protocol unequal_times
python "$EFF4D_ROOT/adapters/train.py" --replay "$EFF4D_ROOT/objects/OBJ_0018/clips/C001/05_reconstruction/E0_S1_12k_v1/checkpoint.pt"
