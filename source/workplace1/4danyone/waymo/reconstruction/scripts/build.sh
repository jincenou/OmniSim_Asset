#!/usr/bin/env bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/scripts/env.sh
R=/mnt/ojc/workplace1/4danyone/waymo/reconstruction
P=/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python
export CUDA_HOME=/mnt/ojc/miniconda3/envs/asset_harvester_2026
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4
export TORCH_EXTENSIONS_DIR="$R/deps/torch_extensions"
trap 'code=$?; printf "%s\n" "$code" > "$R/logs/build.exit"' EXIT
cd "$R/deps"
"$P" -m pip install --no-build-isolation --no-deps ./gsplat ./fused-ssim
"$P" -m pip freeze > "$R/research/requirements.txt"
"$P" /mnt/ojc/workplace1/4danyone/waymo/bridge/backend/src/simple_trainer_freetime_4d_pure_relocation.py --help > "$R/research/trainer-help.txt"
