#!/usr/bin/env bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/waymo/grid4d/scripts/env.sh
cd "$GRID4D_ROOT"
trap 'code=$?; echo "$code" > "$GRID4D_ROOT/logs/build_hash.exit"' EXIT
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export CUDA_VISIBLE_DEVICES=GPU-32040e1a-df6d-0776-5420-fae47bc11248
python -m pip install --no-deps --index-url https://pypi.org/simple cachetools==5.3.3
PYTHONPATH="$GRID4D_ROOT/source" python -c 'import torch, hashencoder; print("hashencoder compiled",torch.__version__)'
python -m pip check
python -m pip freeze > research/environment.lock.txt
git -C source diff > research/source.patch
touch logs/build.complete
