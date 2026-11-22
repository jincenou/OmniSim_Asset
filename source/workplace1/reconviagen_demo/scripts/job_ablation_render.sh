#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=4
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PATH=/home/wmj/miniconda3/envs/sim/bin:$PATH
export CPATH=/home/wmj/miniconda3/envs/splatad/include
export TORCH_EXTENSIONS_DIR=/mnt/ojc/workplace1/reconviagen_demo/cache/torch_extensions
set -e
env/bin/python scripts/prepare_asset.py --case fresh_black_pickup --variant R1 --seed 42
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_orbit.py --ply results/fresh_black_pickup/R1_seed42/asset/canonical.ply --output results/fresh_black_pickup/R1_seed42/asset --length 5.244319285120627
env/bin/python scripts/prepare_asset.py --case fresh_black_pickup --variant R2 --seed 42
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_orbit.py --ply results/fresh_black_pickup/R2_seed42/asset/canonical.ply --output results/fresh_black_pickup/R2_seed42/asset --length 5.244319285120627
env/bin/python scripts/prepare_asset.py --case fresh_black_pickup --variant R --seed 7
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_orbit.py --ply results/fresh_black_pickup/R_seed7/asset/canonical.ply --output results/fresh_black_pickup/R_seed7/asset --length 5.244319285120627
