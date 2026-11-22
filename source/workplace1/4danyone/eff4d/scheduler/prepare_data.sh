#!/bin/bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/eff4d/scheduler/env.sh
trap 'code=$?; echo "$code" > "$EFF4D_ROOT/scheduler/prepare_data.exit"' EXIT
export CUDA_VISIBLE_DEVICES=''
python "$EFF4D_ROOT/adapters/data.py" --object R07 --prepare --initialize S0
python "$EFF4D_ROOT/adapters/data.py" --object R07 --initialize S1
python "$EFF4D_ROOT/adapters/data.py" --object M02 --prepare --initialize S1
