#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python -u scripts/evaluate_asset_metrics.py --case random10_04_white_suv --variants B0 --encoder dino > logs/random10_04_white_suv_B0_dino.log 2>&1
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python -u scripts/evaluate_asset_metrics.py --case random10_09_dark_sedan --variants B0 --encoder dino > logs/random10_09_dark_sedan_B0_dino.log 2>&1
echo 0 > logs/random10_B0_dino_gpu6.exit
