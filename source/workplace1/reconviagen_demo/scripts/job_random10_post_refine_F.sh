#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export PATH=/home/wmj/miniconda3/envs/sim/bin:$PATH CPATH=/home/wmj/miniconda3/envs/splatad/include TORCH_EXTENSIONS_DIR=/mnt/ojc/workplace1/reconviagen_demo/cache/torch_extensions MAX_JOBS=4
env/bin/python scripts/prepare_asset.py --case random10_02_white_pickup --variant F_refine_base
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_orbit.py --ply results/random10_02_white_pickup/F_refine_base_seed42/asset/canonical.ply --output results/random10_02_white_pickup/F_refine_base_seed42/asset --length 5.738485901862793
env/bin/python scripts/prepare_asset.py --case random10_02_white_pickup --variant F_refined
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_orbit.py --ply results/random10_02_white_pickup/F_refined_seed42/asset/canonical.ply --output results/random10_02_white_pickup/F_refined_seed42/asset --length 5.738485901862793
echo 0 > logs/random10_post_refine_F.exit
