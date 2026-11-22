#!/bin/bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/eff4d/scheduler/env.sh
trap 'code=$?; echo "$code" > "$EFF4D_ROOT/scheduler/setup_local.exit"' EXIT
export CUDA_VISIBLE_DEVICES=4
/mnt/ojc/miniconda3/bin/conda create -y --offline --copy --prefix "$EFF4D_ENV" --file "$EFF4D_ROOT/configs/python_explicit.txt"
"$EFF4D_ENV/bin/python" -c 'import sys; print(sys.prefix)'
