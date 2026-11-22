#!/bin/bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/eff4d/scheduler/env.sh
trap 'code=$?; echo "$code" > "$EFF4D_ROOT/scheduler/setup_base.exit"' EXIT
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export CUDA_VISIBLE_DEVICES=4
mkdir -p "$CONDA_PKGS_DIRS"
/mnt/ojc/miniconda3/bin/conda create -y --copy --prefix "$EFF4D_ENV" --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge python=3.10 pip=24.3.1
"$EFF4D_ENV/bin/python" -c 'import sys; print(sys.prefix)'
