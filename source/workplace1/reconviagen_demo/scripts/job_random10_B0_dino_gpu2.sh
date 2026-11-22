#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python -u scripts/evaluate_asset_metrics.py --case random10_02_white_pickup --variants B0 --encoder dino > logs/random10_02_white_pickup_B0_dino.log 2>&1
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python -u scripts/evaluate_asset_metrics.py --case random10_07_grey_minivan --variants B0 --encoder dino > logs/random10_07_grey_minivan_B0_dino.log 2>&1
echo 0 > logs/random10_B0_dino_gpu2.exit
