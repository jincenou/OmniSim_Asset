#!/bin/bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/eff4d/scheduler/env.sh
trap 'code=$?; echo "$code" > "$EFF4D_ROOT/scheduler/build_v2.exit"' EXIT
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export CUDA_VISIBLE_DEVICES=4
python -m pip install --no-deps --index-url https://pypi.org/simple imagesize==1.4.1 kornia==0.6.12 lpips==0.1.4 pytorch-msssim==1.0.0
test -x "$EFF4D_ENV/bin/ninja"
python -m pip install --no-deps --no-build-isolation "$EFF4D_ROOT/repos/4d-gaussian-splatting/simple-knn"
cd "$EFF4D_ROOT/repos/4d-gaussian-splatting"
python -c 'import torch; print(torch.__version__,torch.cuda.get_device_properties(0)); import gaussian_renderer; print("4D rasterizer loaded from generation source")'
