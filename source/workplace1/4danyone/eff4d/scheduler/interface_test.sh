#!/bin/bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/eff4d/scheduler/env.sh
trap 'code=$?; echo "$code" > "$EFF4D_ROOT/scheduler/interface_test.exit"' EXIT
export CUDA_VISIBLE_DEVICES=4
python "$EFF4D_ROOT/adapters/train.py" --manifest "$EFF4D_ROOT/research/interface_test/short_clip/manifest.json" --protocol seven_view_short --variant E0 --run-id interface20_v1 --steps 20
