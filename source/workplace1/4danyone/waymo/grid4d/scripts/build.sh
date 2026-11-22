#!/usr/bin/env bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/waymo/grid4d/scripts/env.sh
cd "$GRID4D_ROOT"
trap 'code=$?; echo "$code" > "$GRID4D_ROOT/logs/build.exit"' EXIT
export CUDA_VISIBLE_DEVICES=GPU-32040e1a-df6d-0776-5420-fae47bc11248
python -m pip install --no-deps --no-build-isolation ./source/submodules/depth-diff-gaussian-rasterization
python -m pip install --no-deps --no-build-isolation ./source/submodules/simple-knn
PYTHONPATH="$GRID4D_ROOT/source" python -c 'import torch, hashencoder; print("hashencoder compiled",torch.__version__)'
python -m pip check
python -m pip freeze > research/environment.lock.txt
git -C source diff > research/source.patch
touch logs/build.complete
