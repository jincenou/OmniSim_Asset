#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export PATH=/home/wmj/miniconda3/envs/sim/bin:$PATH
export CUDA_HOME=/home/wmj/miniconda3/envs/sim
export CPATH=/home/wmj/miniconda3/envs/splatad/include
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=4
export PIP_CACHE_DIR=/mnt/ojc/workplace1/reconviagen_demo/cache/pip
curl -L --fail --connect-timeout 20 --max-time 300 https://codeload.github.com/autonomousvision/mip-splatting/tar.gz/refs/heads/main -o cache/mip-splatting.tar.gz
mkdir -p vendor/mip-splatting
tar -xzf cache/mip-splatting.tar.gz -C vendor/mip-splatting --strip-components=1
env/bin/pip install --no-build-isolation --no-deps --no-cache-dir --force-reinstall vendor/mip-splatting/submodules/diff-gaussian-rasterization
echo 0 > logs/build_raster.exit
