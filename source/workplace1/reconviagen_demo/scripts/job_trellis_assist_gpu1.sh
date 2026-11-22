#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PATH=/home/wmj/miniconda3/envs/sim/bin:$PATH
export CPATH=/home/wmj/miniconda3/envs/splatad/include
export TORCH_EXTENSIONS_DIR=/mnt/ojc/workplace1/reconviagen_demo/cache/torch_extensions
export MAX_JOBS=4
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/baseline.py --case fresh_black_pickup --real-count 4 --seed 7 > logs/fresh_black_pickup_B0_real4_seed7_inference.log 2>&1
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/prepare_asset.py --case fresh_black_pickup --variant B0_real4 --seed 7 > logs/fresh_black_pickup_B0_real4_seed7_export.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_orbit.py --ply results/fresh_black_pickup/B0_real4_seed7/asset/canonical.ply --output results/fresh_black_pickup/B0_real4_seed7/asset --length 5.244319285120627 --distance 11 > logs/fresh_black_pickup_B0_real4_seed7_render.log 2>&1
