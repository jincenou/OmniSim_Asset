#!/usr/bin/env bash
set -euo pipefail
ALT_ROOT=/mnt/ojc/workplace1/4danyone/waymo/round2/alternatives/4dgaussians
ALT_ENV=/mnt/ojc/miniconda3/envs/4dgaussians
mkdir -p "$ALT_ROOT/cache/pip" "$ALT_ROOT/cache/conda" "$ALT_ROOT/tmp" "$ALT_ROOT/reports"
export PIP_CACHE_DIR="$ALT_ROOT/cache/pip"
export CONDA_PKGS_DIRS="$ALT_ROOT/cache/conda"
export TMPDIR="$ALT_ROOT/tmp"
export PYTHONNOUSERSITE=1
unset HTTPS_PROXY HTTP_PROXY ALL_PROXY https_proxy http_proxy all_proxy
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=http.proxy GIT_CONFIG_VALUE_0=''
trap 'code=$?; printf "%s\n" "$code" > "$ALT_ROOT/setup_retry.exit"' EXIT
test ! -e "$ALT_ENV"
/mnt/ojc/miniconda3/bin/conda create --offline --copy --yes --prefix "$ALT_ENV" --clone /mnt/ojc/miniconda3/envs/idol
export PATH="$ALT_ENV/bin:$PATH"
export CUDA_HOME="$ALT_ENV"
export TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4 OMP_NUM_THREADS=4
export TORCH_EXTENSIONS_DIR="$ALT_ROOT/cache/extensions"
python -c 'import sys,torch; assert sys.prefix=="/mnt/ojc/miniconda3/envs/4dgaussians"; assert torch.__version__=="2.3.1+cu118"; print(sys.prefix,torch.__version__)'
git -C "$ALT_ROOT/repo" submodule update --init --recursive submodules/depth-diff-gaussian-rasterization submodules/simple-knn
test "$(git -C "$ALT_ROOT/repo/submodules/depth-diff-gaussian-rasterization" rev-parse HEAD)" = e49506654e8e11ed8a62d22bcb693e943fdecacf
test "$(git -C "$ALT_ROOT/repo/submodules/simple-knn" rev-parse HEAD)" = 44f764299fa305faf6ec5ebd99939e0508331503
python -m pip install 'lpips==0.1.4' 'pytorch-msssim==1.0.0' 'mmcv==1.6.0' 'tensorboard==2.20.0'
python -m pip install --no-build-isolation --no-deps --force-reinstall "$ALT_ROOT/repo/submodules/depth-diff-gaussian-rasterization" "$ALT_ROOT/repo/submodules/simple-knn"
python -m pip check
python -m pip freeze > "$ALT_ROOT/reports/requirements.lock.txt"
git -C "$ALT_ROOT/repo" submodule status --recursive > "$ALT_ROOT/reports/submodules.txt"
printf '%s\n' environment_packages_installed_runtime_validation_pending > "$ALT_ROOT/setup_retry.state"
