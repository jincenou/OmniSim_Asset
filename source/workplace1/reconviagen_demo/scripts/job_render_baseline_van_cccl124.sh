#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
export PATH=/home/wmj/miniconda3/envs/sim/bin:$PATH
export CPATH=/home/wmj/miniconda3/envs/splatad/include
export TORCH_EXTENSIONS_DIR=/mnt/ojc/workplace1/reconviagen_demo/cache/torch_extensions
export MAX_JOBS=4
export CUDA_VISIBLE_DEVICES=0
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_orbit.py --ply results/service_van/B0_seed42/asset/canonical.ply --output results/service_van/B0_seed42/asset --length 5.690076076823228 > logs/render_baseline_van_cccl124.log 2>&1
code=$?
printf "%s\n" "$code" > logs/render_baseline_van_cccl124.exit
exit "$code"
