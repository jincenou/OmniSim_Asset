#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python -u scripts/evaluate_asset_metrics.py --case fresh_silver_suv --encoder dino > logs/fresh_silver_suv_all_dino.log 2>&1
echo 0 > logs/fresh_silver_suv_all_dino.exit
