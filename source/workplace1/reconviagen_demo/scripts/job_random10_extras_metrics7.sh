#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
while [ ! -f results/random10_02_white_pickup/R_seed7/evaluation/render_audit.json ]; do sleep 5; done
while [ ! -f results/random10_02_white_pickup/B0_real1_seed7/evaluation/render_audit.json ]; do sleep 5; done
while [ ! -f results/random10_02_white_pickup/B0_real4_seed7/evaluation/render_audit.json ]; do sleep 5; done
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/evaluate_asset_metrics.py --case random10_02_white_pickup --seed 7 --variants R B0_real1 B0_real4 --encoder clip > logs/random10_extras_seed7_clip.log 2>&1
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/evaluate_asset_metrics.py --case random10_02_white_pickup --seed 7 --variants R B0_real1 B0_real4 --encoder dino > logs/random10_extras_seed7_dino.log 2>&1
echo 0 > logs/random10_extras_seed7_metrics.exit
