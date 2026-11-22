#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=5
export PATH=/home/wmj/miniconda3/envs/sim/bin:$PATH
export CPATH=/home/wmj/miniconda3/envs/splatad/include
export TORCH_EXTENSIONS_DIR=/mnt/ojc/workplace1/reconviagen_demo/cache/torch_extensions
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MAX_JOBS=4
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/prepare_asset.py --case fresh_black_pickup --variant F_refined
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_orbit.py --ply results/fresh_black_pickup/F_refined_seed42/asset/canonical.ply --output results/fresh_black_pickup/F_refined_seed42/asset --length 5.244319285120627 --distance 11
